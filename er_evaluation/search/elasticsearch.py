from er_evaluation.search.http_requests import http_post_request


class ElasticSearch:
    """
    ElasticSearch client for user-friendly search queries and aggregation.

    Note:
        The class supports fields with up to one nested level. The use of higher levels of nesting has not yet been tested.
    """
    
    def __init__(self, base_url, api_key=None):
        """
        Initialize the ElasticSearch client with a base URL and an optional API key.

        Args:
            base_url: The base URL of the Elasticsearch server.
            api_key: The API key for authentication (optional).
        """
        self.base_url = base_url
        self.api_key = api_key

    @staticmethod
    def _build_headers(api_key=None):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"ApiKey {api_key}"
        return headers

    @staticmethod
    def _process_query(user_query, fields, fuzziness=2):
        """
        Process the user's query and build a search query for Elasticsearch.

        Args:
            user_query: The user's query string.
            fields: A list of fields to search in.
            fuzziness: The fuzziness level for matching (optional, default: 2).
        Returns:
            A dictionary representing the Elasticsearch search query.
        """

        def create_nested_query(field, full_field_path, query):
            """Helper function to create nested queries"""
            path = field.split(".")
            if len(path) > 1:
                return {"nested": {"path": path[0], "query": create_nested_query(".".join(path[1:]), full_field_path, query)}}
            else:
                return {"match": {full_field_path: query}}

        must_clauses = []
        for field in fields:
            query = {"query": user_query, "fuzziness": fuzziness}
            must_clauses.append(create_nested_query(field, field, query))

        return {"query": {"bool": {"should": must_clauses}}}

    @staticmethod
    def _process_aggregations(agg_fields, agg_size=10000, _source=None, top_hits_size=5):
        """
        Process the aggregation fields and build an aggregation query for Elasticsearch.

        Args:
            agg_fields: A list of fields to aggregate on.
            agg_size: The maximum number of aggregation entries (optional, default: 10000).
            _source: A list of fields to return for each top hit (optional).
            top_hits_size: The number of top hits to include for each bucket (optional, default: 5).

        Returns:
            A dictionary representing the Elasticsearch aggregation query.
        """
        def create_nested_agg(field, full_field_path, size, _source, top_hits_size):
            """Helper function to create nested aggregations"""
            path = field.split('.')
            if len(path) > 1:
                return {
                    full_field_path: {
                        "nested": {"path": '.'.join(path[:-1])},
                        "aggs": create_nested_agg('.'.join(path[1:]), full_field_path, size, _source, top_hits_size)
                    }
                }
            else:
                aggs = {
                    full_field_path+"_inner": {
                        "terms": {"field": full_field_path, "size": size},
                        "aggs": {}
                    }
                }
                if _source is not None:
                    aggs[full_field_path+"_inner"]["aggs"]["top_hits"] = {
                        "top_hits": {
                            "_source": {
                                "includes": _source
                            },
                            "size": top_hits_size
                        }
                    }

                return aggs

        agg_query = {}
        for agg_field in agg_fields:
            agg_query.update(create_nested_agg(agg_field, agg_field, agg_size, _source, top_hits_size))

        return agg_query

    def search(
        self,
        user_query,
        index,
        fields,
        fuzziness=2,
        agg_fields=None,
        agg_size=10000,
        agg_source=None,
        agg_source_top_hits=5,
        source=None,
        timeout=5,
        max_retries=1,
    ):
        """
        Perform a search using the user's query on the specified index and fields.

        Args:
            user_query: The user's query string.
            index: The index to search in.
            fields: A list of fields to search in. 
            fuzziness: The fuzziness level for matching (optional, default: 2).
            agg_fields: A list of fields to aggregate on (optional).
            agg_size: The maximum number of aggregation results to return (optional, default: 10000).
            agg_source: A list of fields to return for each top hit in the aggregations (optional).
            agg_source_top_hits: The number of top hits to include for each bucket in the aggregations (optional, default: 5).
            source: A list of fields to include in the response (optional).
            timeout: The timeout for the request in seconds (optional, default: 5).
            max_retries: The maximum number of retries for the request (optional, default: 1).

        Returns:
            A JSON object containing the search results.

        Raises:
            HTTPError: If the request fails.
        """
        search_query = ElasticSearch._process_query(user_query, fields, fuzziness)
        if source is not None:
            search_query["_source"] = source

        if agg_fields:
            search_query["aggs"] = ElasticSearch._process_aggregations(agg_fields, agg_size, _source=agg_source, top_hits_size=agg_source_top_hits)

        headers = ElasticSearch._build_headers(self.api_key)
        return http_post_request(
            f"{self.base_url}/{index}/_search", headers, search_query, timeout=timeout, max_retries=max_retries
        )
