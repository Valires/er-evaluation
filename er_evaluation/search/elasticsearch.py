import requests


class ElasticSearch:
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
    def process_query(user_query, fields, fuzziness=2):
        """
        Process the user's query and build a search query for Elasticsearch.

        Args:
            user_query: The user's query string.
            fields: A list of fields to search in. This supports nested field with at most one nested level.
            fuzziness: The fuzziness level for matching (optional, default: 2).
        Returns:
            A dictionary representing the Elasticsearch search query.
        """
        must_clauses = []
        for field in fields:
            if "." in field:
                path = field.split(".")[0]
                must_clauses.append(
                    {
                        "nested": {
                            "path": path,
                            "query": {"match": {field: {"query": user_query, "fuzziness": fuzziness}}},
                        }
                    }
                )
            else:
                must_clauses.append({"match": {field: {"query": user_query, "fuzziness": fuzziness}}})

        return {"query": {"bool": {"should": must_clauses}}}

    def search(self, user_query, index, fields, fuzziness=2, agg_fields=None, agg_size=10000, source=None):
        """
        Perform a search using the user's query on the specified index and fields.

        Args:
            user_query: The user's query string.
            index: The index to search in.
            fields: A list of fields to search in. This supports nested field with at most one nested level.
            fuzziness: The fuzziness level for matching (optional, default: 2).
            source: A list of fields to include in the response (optional).

        Returns:
            A JSON object containing the search results.
        """
        search_query = ElasticSearch.process_query(user_query, fields, fuzziness)
        if source is not None:
            search_query["_source"] = source

        if agg_fields:
            search_query["aggs"] = {}
            for agg_field in agg_fields:
                if "." in agg_field:
                    nested_path = agg_field.rsplit(".", 1)[0]
                    search_query["aggs"][agg_field] = {
                        "nested": {"path": nested_path},
                        "aggs": {agg_field + "_inner": {"terms": {"field": agg_field, "size": agg_size}}},
                    }
                else:
                    search_query["aggs"][agg_field] = {"terms": {"field": agg_field, "size": agg_size}}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"
        response = requests.post(f"{self.base_url}/{index}/_search", json=search_query, headers=headers)
        return response.json()
