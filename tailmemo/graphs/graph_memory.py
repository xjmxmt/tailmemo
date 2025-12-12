from loguru import logger

from tailmemo.memory.utils import format_entities, sanitize_relationship_for_cypher

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    raise ImportError("langchain_neo4j is not installed. Please install it using pip install langchain-neo4j")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from tailmemo.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from tailmemo.configs.prompts import NODE_EXTRACTION_PROMPTS, EXTRACT_RELATIONS_PROMPT, get_delete_messages
from tailmemo.utils.factory import EmbedderFactory, LlmFactory


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        self.graph = Neo4jGraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
            self.config.graph_store.config.database,
            refresh_schema=False,
            driver_config={"notifications_min_severity": "OFF"},
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config
        )
        self.node_label = ":`__Entity__`" if self.config.graph_store.config.base_label else ""

        if self.config.graph_store.config.base_label:
            # Safely add user_id index
            try:
                self.graph.query(f"CREATE INDEX entity_single IF NOT EXISTS FOR (n {self.node_label}) ON (n.user_id)")
            except Exception:
                pass
            try:  # Safely try to add composite index (Enterprise only)
                self.graph.query(
                    f"CREATE INDEX entity_composite IF NOT EXISTS FOR (n {self.node_label}) ON (n.name, n.user_id)"
                )
            except Exception:
                pass

        # Default to openai if no specific provider is configured
        self.llm_provider = "openai"
        if self.config.llm and self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store and self.config.graph_store.llm and self.config.graph_store.llm.provider:
            self.llm_provider = self.config.graph_store.llm.provider

        # Get LLM config with proper null checks
        llm_config = None
        if self.config.graph_store and self.config.graph_store.llm and hasattr(self.config.graph_store.llm, "config"):
            llm_config = self.config.graph_store.llm.config
        elif hasattr(self.config.llm, "config"):
            llm_config = self.config.llm.config
        self.llm = LlmFactory.create(self.llm_provider, llm_config)
        self.user_id = None
        # Use threshold from graph_store config, default to 0.9
        self.threshold = self.config.graph_store.threshold if hasattr(self.config.graph_store, 'threshold') else 0.9

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        entity_info_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_info_map)
        search_output = self._search_graph_db(node_list=list(entity_info_map.keys()), filters=filters, entity_info_map=entity_info_map)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)

        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support
        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_info_map)

        logger.debug(f"deleted_entities: {deleted_entities},"
                     f"added_entities: {added_entities}")
        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        entity_info_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_info_map.keys()), filters=filters, entity_info_map=entity_info_map)

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def delete_all(self, filters):
        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        cypher = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.
         Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """
        params = {"user_id": filters["user_id"], "limit": limit}

        # Build node properties based on filters
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
            params["run_id"] = filters["run_id"]
        node_props_str = ", ".join(node_props)

        query = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})-[r]->(m {self.node_label} {{{node_props_str}}})
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query.
        
        Returns:
            dict: A dictionary mapping entity names to their metadata:
                {
                    "entity_name": {
                        "entity_type": str,
                        "description": str,  # Context-aware description for enhanced embedding
                        "aliases": list[str]  # Alternative names for entity linking
                    }
                }
        """
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": NODE_EXTRACTION_PROMPTS,
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_info_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_name = item["entity"].lower().replace(" ", "_")
                    entity_info_map[entity_name] = {
                        "entity_type": item.get("entity_type", "unknown").lower().replace(" ", "_"),
                        "description": item.get("description", ""),
                        "aliases": [alias.lower().replace(" ", "_") for alias in item.get("aliases", [])],
                    }
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        logger.debug(f"Entity info map: {entity_info_map}\n search_results={search_results}")
        return entity_info_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_info_map):
        """Establish relations among the extracted nodes."""

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT
            # Add the custom prompt line if configured
            system_content = system_content.replace("CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"List of entities: {list(entity_info_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _compute_enhanced_embedding(self, entity_name, entity_info=None):
        """Compute embedding using entity name enhanced with context description.
        
        Args:
            entity_name: The name of the entity
            entity_info: Optional dict containing 'description' and 'aliases'
            
        Returns:
            The embedding vector for the enhanced entity representation
        """
        if entity_info and entity_info.get("description"):
            # Use context-enhanced embedding: "entity_name: description"
            enhanced_text = f"{entity_name}: {entity_info['description']}"
        else:
            enhanced_text = entity_name
        return self.embedding_model.embed(enhanced_text)

    def _search_graph_db(self, node_list, filters, limit=100, entity_info_map=None):
        """Search similar nodes among and their respective incoming and outgoing relations.
        
        Uses context-enhanced embeddings and supports alias matching for better entity linking.
        """
        result_relations = []
        entity_info_map = entity_info_map or {}

        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        # Collect all names to search (primary names + aliases)
        names_to_search = []
        for node in node_list:
            entity_info = entity_info_map.get(node, {})
            names_to_search.append((node, entity_info))
            # Also search for aliases
            for alias in entity_info.get("aliases", []):
                if alias != node:  # Avoid duplicates
                    names_to_search.append((alias, entity_info))

        seen_relations = set()  # Track unique relations to avoid duplicates

        for search_name, entity_info in names_to_search:
            # Use context-enhanced embedding
            n_embedding = self._compute_enhanced_embedding(search_name, entity_info)

            # Query with both vector similarity and alias matching
            cypher_query = f"""
            MATCH (n {self.node_label} {{{node_props_str}}})
            WHERE n.embedding IS NOT NULL
            WITH n, 
                 round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS embedding_similarity,
                 CASE 
                     WHEN n.name = $search_name THEN 1.0
                     WHEN $search_name IN coalesce(n.aliases, []) THEN 0.95
                     ELSE 0.0 
                 END AS alias_match_score
            WITH n, 
                 CASE 
                     WHEN alias_match_score > 0 THEN alias_match_score
                     ELSE embedding_similarity
                 END AS similarity
            WHERE similarity >= $threshold
            CALL {{
                WITH n
                MATCH (n)-[r]->(m {self.node_label} {{{node_props_str}}})
                RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id
                UNION
                WITH n  
                MATCH (n)<-[r]-(m {self.node_label} {{{node_props_str}}})
                RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id
            }}
            WITH distinct source, source_id, relationship, relation_id, destination, destination_id, similarity
            RETURN source, source_id, relationship, relation_id, destination, destination_id, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """

            params = {
                "n_embedding": n_embedding,
                "search_name": search_name,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
            }
            if filters.get("agent_id"):
                params["agent_id"] = filters["agent_id"]
            if filters.get("run_id"):
                params["run_id"] = filters["run_id"]

            ans = self.graph.query(cypher_query, params=params)
            
            # Deduplicate results based on relation_id
            for item in ans:
                relation_key = item.get("relation_id")
                if relation_key and relation_key not in seen_relations:
                    seen_relations.add(relation_key)
                    result_relations.append(item)

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build the agent filter for the query

            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }

            if agent_id:
                params["agent_id"] = agent_id
            if run_id:
                params["run_id"] = run_id

            # Build node properties for filtering
            source_props = ["name: $source_name", "user_id: $user_id"]
            dest_props = ["name: $dest_name", "user_id: $user_id"]
            if agent_id:
                source_props.append("agent_id: $agent_id")
                dest_props.append("agent_id: $agent_id")
            if run_id:
                source_props.append("run_id: $run_id")
                dest_props.append("run_id: $run_id")
            source_props_str = ", ".join(source_props)
            dest_props_str = ", ".join(dest_props)

            # Delete the specific relationship between nodes
            cypher = f"""
            MATCH (n {self.node_label} {{{source_props_str}}})
            -[r:{relationship}]->
            (m {self.node_label} {{{dest_props_str}}})

            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """

            result = self.graph.query(cypher, params=params)
            results.append(result)

        return results

    def _add_entities(self, to_be_added, filters, entity_info_map):
        """Add the new entities to the graph. Merge the nodes if they already exist.
        
        Stores entity aliases and uses context-enhanced embeddings for better entity linking.
        """
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []
        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Get entity info (with description and aliases)
            source_info = entity_info_map.get(source, {})
            destination_info = entity_info_map.get(destination, {})

            # types (now from entity_info_map)
            source_type = source_info.get("entity_type", "__User__")
            source_label = self.node_label if self.node_label else f":`{source_type}`"
            source_extra_set = f", source:`{source_type}`" if self.node_label else ""
            destination_type = destination_info.get("entity_type", "__User__")
            destination_label = self.node_label if self.node_label else f":`{destination_type}`"
            destination_extra_set = f", destination:`{destination_type}`" if self.node_label else ""

            # Get aliases and descriptions
            source_aliases = source_info.get("aliases", [])
            source_description = source_info.get("description", "")
            dest_aliases = destination_info.get("aliases", [])
            dest_description = destination_info.get("description", "")

            # Use context-enhanced embeddings
            source_embedding = self._compute_enhanced_embedding(source, source_info)
            dest_embedding = self._compute_enhanced_embedding(destination, destination_info)

            # search for the nodes with the closest embeddings (now with alias support)
            source_node_search_result = self._search_source_node(
                source_embedding, filters, threshold=self.threshold,
                entity_name=source, aliases=source_aliases
            )
            destination_node_search_result = self._search_destination_node(
                dest_embedding, filters, threshold=self.threshold,
                entity_name=destination, aliases=dest_aliases
            )

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                # Build destination MERGE properties
                merge_props = ["name: $destination_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1,
                    source.aliases = reduce(seen = [], x IN coalesce(source.aliases, []) + $source_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    source.description = CASE WHEN $source_description <> '' THEN $source_description ELSE source.description END
                WITH source
                MERGE (destination {destination_label} {{{merge_props_str}}})
                ON CREATE SET
                    destination.created = timestamp(),
                    destination.mentions = 1,
                    destination.aliases = $dest_aliases,
                    destination.description = $dest_description
                    {destination_extra_set}
                ON MATCH SET
                    destination.mentions = coalesce(destination.mentions, 0) + 1,
                    destination.aliases = reduce(seen = [], x IN coalesce(destination.aliases, []) + $dest_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    destination.description = CASE WHEN $dest_description <> '' THEN $dest_description ELSE destination.description END
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $destination_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "source_aliases": source_aliases,
                    "source_description": source_description,
                    "dest_aliases": dest_aliases,
                    "dest_description": dest_description,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif destination_node_search_result and not source_node_search_result:
                # Build source MERGE properties
                merge_props = ["name: $source_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1,
                    destination.aliases = reduce(seen = [], x IN coalesce(destination.aliases, []) + $dest_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    destination.description = CASE WHEN $dest_description <> '' THEN $dest_description ELSE destination.description END
                WITH destination
                MERGE (source {source_label} {{{merge_props_str}}})
                ON CREATE SET
                    source.created = timestamp(),
                    source.mentions = 1,
                    source.aliases = $source_aliases,
                    source.description = $source_description
                    {source_extra_set}
                ON MATCH SET
                    source.mentions = coalesce(source.mentions, 0) + 1,
                    source.aliases = reduce(seen = [], x IN coalesce(source.aliases, []) + $source_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    source.description = CASE WHEN $source_description <> '' THEN $source_description ELSE source.description END
                WITH source, destination
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "source_aliases": source_aliases,
                    "source_description": source_description,
                    "dest_aliases": dest_aliases,
                    "dest_description": dest_description,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif source_node_search_result and destination_node_search_result:
                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1,
                    source.aliases = reduce(seen = [], x IN coalesce(source.aliases, []) + $source_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    source.description = CASE WHEN $source_description <> '' THEN $source_description ELSE source.description END
                WITH source
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1,
                    destination.aliases = reduce(seen = [], x IN coalesce(destination.aliases, []) + $dest_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                    destination.description = CASE WHEN $dest_description <> '' THEN $dest_description ELSE destination.description END
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1
                ON MATCH SET r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "source_aliases": source_aliases,
                    "source_description": source_description,
                    "dest_aliases": dest_aliases,
                    "dest_description": dest_description,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            else:
                # Build dynamic MERGE props for both source and destination
                source_props = ["name: $source_name", "user_id: $user_id"]
                dest_props = ["name: $dest_name", "user_id: $user_id"]
                if agent_id:
                    source_props.append("agent_id: $agent_id")
                    dest_props.append("agent_id: $agent_id")
                if run_id:
                    source_props.append("run_id: $run_id")
                    dest_props.append("run_id: $run_id")
                source_props_str = ", ".join(source_props)
                dest_props_str = ", ".join(dest_props)

                cypher = f"""
                MERGE (source {source_label} {{{source_props_str}}})
                ON CREATE SET source.created = timestamp(),
                            source.mentions = 1,
                            source.aliases = $source_aliases,
                            source.description = $source_description
                            {source_extra_set}
                ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1,
                            source.aliases = reduce(seen = [], x IN coalesce(source.aliases, []) + $source_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                            source.description = CASE WHEN $source_description <> '' THEN $source_description ELSE source.description END
                WITH source
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source
                MERGE (destination {destination_label} {{{dest_props_str}}})
                ON CREATE SET destination.created = timestamp(),
                            destination.mentions = 1,
                            destination.aliases = $dest_aliases,
                            destination.description = $dest_description
                            {destination_extra_set}
                ON MATCH SET destination.mentions = coalesce(destination.mentions, 0) + 1,
                            destination.aliases = reduce(seen = [], x IN coalesce(destination.aliases, []) + $dest_aliases | CASE WHEN x IS NOT NULL AND NOT x IN seen THEN seen + [x] ELSE seen END),
                            destination.description = CASE WHEN $dest_description <> '' THEN $dest_description ELSE destination.description END
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $dest_embedding)
                WITH source, destination
                MERGE (source)-[rel:{relationship}]->(destination)
                ON CREATE SET rel.created = timestamp(), rel.mentions = 1
                ON MATCH SET rel.mentions = coalesce(rel.mentions, 0) + 1
                RETURN source.name AS source, type(rel) AS relationship, destination.name AS target
                """

                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "source_aliases": source_aliases,
                    "source_description": source_description,
                    "dest_aliases": dest_aliases,
                    "dest_description": dest_description,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id
            result = self.graph.query(cypher, params=params)
            logger.debug((f"Added relationships: {result}"))
            results.append(result)
        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            # Use the sanitization function for relationships to handle special characters
            item["relationship"] = sanitize_relationship_for_cypher(item["relationship"].lower().replace(" ", "_"))
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9, entity_name=None, aliases=None):
        """Search for existing source node by name/alias matching first, then by embedding similarity.
        
        Logic:
            1. First match entity_name and aliases against node's name and aliases
            2. Only if alias_match_score > 0 (name/alias match found), calculate embedding_similarity
            3. Return the node with highest embedding_similarity among matched nodes
            4. If no name/alias match, return empty (don't fallback to pure embedding match)
        
        Args:
            source_embedding: The embedding vector of the source entity
            filters: Filter conditions (user_id, agent_id, run_id)
            threshold: Similarity threshold for embedding matching
            entity_name: The name of the entity to match against node names and aliases
            aliases: List of aliases to check for matches
        """
        # Build WHERE conditions
        where_conditions = ["source_candidate.embedding IS NOT NULL", "source_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("source_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("source_candidate.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        # Step 1: Match by name/alias first
        # Step 2: Only calculate embedding similarity for matched nodes
        # Step 3: Return the one with highest embedding similarity
        cypher = f"""
            MATCH (source_candidate {self.node_label})
            WHERE {where_clause}

            // First, calculate alias match score (name/alias matching)
            WITH source_candidate,
                CASE 
                    WHEN $entity_name IS NOT NULL AND source_candidate.name = $entity_name THEN 1.0
                    WHEN $entity_name IS NOT NULL AND $entity_name IN coalesce(source_candidate.aliases, []) THEN 0.98
                    WHEN $aliases IS NOT NULL AND size([a IN $aliases WHERE a IN coalesce(source_candidate.aliases, []) OR a = source_candidate.name]) > 0 THEN 0.95
                    ELSE 0.0
                END AS alias_match_score
            
            // Only proceed if there's a name/alias match
            WHERE alias_match_score > 0
            
            // Now calculate embedding similarity for matched nodes
            WITH source_candidate, alias_match_score,
                round(2 * vector.similarity.cosine(source_candidate.embedding, $source_embedding) - 1, 4) AS embedding_similarity
            
            // Filter by embedding threshold and sort by embedding similarity
            WHERE embedding_similarity >= $threshold
            ORDER BY embedding_similarity DESC
            LIMIT 1

            RETURN elementId(source_candidate)
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
            "entity_name": entity_name,
            "aliases": aliases or [],
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9, entity_name=None, aliases=None):
        """Search for existing destination node by name/alias matching first, then by embedding similarity.
        
        Logic:
            1. First match entity_name and aliases against node's name and aliases
            2. Only if alias_match_score > 0 (name/alias match found), calculate embedding_similarity
            3. Return the node with highest embedding_similarity among matched nodes
            4. If no name/alias match, return empty (don't fallback to pure embedding match)
        
        Args:
            destination_embedding: The embedding vector of the destination entity
            filters: Filter conditions (user_id, agent_id, run_id)
            threshold: Similarity threshold for embedding matching
            entity_name: The name of the entity to match against node names and aliases
            aliases: List of aliases to check for matches
        """
        # Build WHERE conditions
        where_conditions = ["destination_candidate.embedding IS NOT NULL", "destination_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("destination_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("destination_candidate.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        # Step 1: Match by name/alias first
        # Step 2: Only calculate embedding similarity for matched nodes
        # Step 3: Return the one with highest embedding similarity
        cypher = f"""
            MATCH (destination_candidate {self.node_label})
            WHERE {where_clause}

            // First, calculate alias match score (name/alias matching)
            WITH destination_candidate,
                CASE 
                    WHEN $entity_name IS NOT NULL AND destination_candidate.name = $entity_name THEN 1.0
                    WHEN $entity_name IS NOT NULL AND $entity_name IN coalesce(destination_candidate.aliases, []) THEN 0.98
                    WHEN $aliases IS NOT NULL AND size([a IN $aliases WHERE a IN coalesce(destination_candidate.aliases, []) OR a = destination_candidate.name]) > 0 THEN 0.95
                    ELSE 0.0
                END AS alias_match_score
            
            // Only proceed if there's a name/alias match
            WHERE alias_match_score > 0
            
            // Now calculate embedding similarity for matched nodes
            WITH destination_candidate, alias_match_score,
                round(2 * vector.similarity.cosine(destination_candidate.embedding, $destination_embedding) - 1, 4) AS embedding_similarity
            
            // Filter by embedding threshold and sort by embedding similarity
            WHERE embedding_similarity >= $threshold
            ORDER BY embedding_similarity DESC
            LIMIT 1

            RETURN elementId(destination_candidate)
            """

        params = {
            "destination_embedding": destination_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
            "entity_name": entity_name,
            "aliases": aliases or [],
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params=params)
        return result

    # Reset is not defined in base.py
    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = """
        MATCH (n) DETACH DELETE n
        """
        return self.graph.query(cypher_query)
