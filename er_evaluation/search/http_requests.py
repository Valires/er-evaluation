import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def create_http_session(max_retries, backoff_factor, status_forcelist):
    """
    Create an HTTP session with a custom retry strategy.
    
    Args:
        max_retries: The maximum number of retries for the request.
        backoff_factor: A factor to use for the backoff algorithm.
        status_forcelist: A tuple of HTTP status codes that should trigger a retry.

    Returns:
        A requests.Session object with the configured retry strategy.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def http_post_request(
    url, headers, json, timeout=10, max_retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)
):
    """
    Perform an HTTP POST request with a custom retry strategy.

    Args:
        url: The URL to send the request to.
        headers: A dictionary of headers to include in the request.
        json: The JSON data to send in the request body.
        timeout: The timeout for the request in seconds (optional, default: 10).
        max_retries: The maximum number of retries for the request (optional, default: 3).
        backoff_factor: A factor to use for the backoff algorithm (optional, default: 0.3).
        status_forcelist: A tuple of HTTP status codes that should trigger a retry (optional, default: (500, 502, 504)).

    Returns:
        A JSON object containing the response data.

    Raises:
        requests.exceptions.HTTPError: If the request fails after the specified number of retries.
    """
    session = create_http_session(max_retries, backoff_factor, status_forcelist)
    try:
        response = session.post(url, json=json, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    finally:
        session.close()
