"""
HTMX Helper Utilities for FastAPI

Provides utility functions for working with HTMX in FastAPI applications,
including request detection, response headers, and common patterns.
"""

from fastapi import Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from typing import Optional, Dict, Any
import json


def is_htmx_request(request: Request) -> bool:
    """
    Check if the request was made by HTMX.

    HTMX adds the 'HX-Request' header to all requests it makes.

    Args:
        request: FastAPI Request object

    Returns:
        bool: True if request is from HTMX
    """
    return request.headers.get("HX-Request", "false") == "true"


def htmx_redirect(url: str, push_url: bool = True) -> Response:
    """
    Create an HTMX-compatible redirect response.

    Instead of a standard HTTP redirect, this returns a 200 response
    with the HX-Redirect header, which HTMX will follow client-side.

    Args:
        url: URL to redirect to
        push_url: Whether to push the URL to browser history

    Returns:
        Response with HX-Redirect header
    """
    headers = {
        "HX-Redirect": url
    }
    if not push_url:
        headers["HX-Push-Url"] = "false"

    return Response(status_code=200, headers=headers)


def htmx_trigger(
    response: Response,
    event_name: str,
    detail: Optional[Dict[str, Any]] = None,
    after: str = "receive"
) -> Response:
    """
    Add an HX-Trigger header to trigger client-side events.

    Args:
        response: Response to add trigger to
        event_name: Name of the event to trigger
        detail: Optional event detail data
        after: When to trigger ('receive', 'settle', 'swap')

    Returns:
        Response with HX-Trigger header added
    """
    header_name = f"HX-Trigger-{after.capitalize()}" if after != "receive" else "HX-Trigger"

    if detail:
        # Complex trigger with data
        trigger_value = json.dumps({event_name: detail})
    else:
        # Simple trigger
        trigger_value = event_name

    response.headers[header_name] = trigger_value
    return response


def htmx_refresh(response: Response) -> Response:
    """
    Tell HTMX to refresh the current page.

    Args:
        response: Response to add refresh header to

    Returns:
        Response with HX-Refresh header
    """
    response.headers["HX-Refresh"] = "true"
    return response


def htmx_reswap(response: Response, swap_method: str) -> Response:
    """
    Override the swap method for this response.

    Args:
        response: Response to modify
        swap_method: Swap method (innerHTML, outerHTML, beforebegin, etc.)

    Returns:
        Response with HX-Reswap header
    """
    response.headers["HX-Reswap"] = swap_method
    return response


def htmx_retarget(response: Response, target: str) -> Response:
    """
    Override the target element for this response.

    Args:
        response: Response to modify
        target: CSS selector for new target

    Returns:
        Response with HX-Retarget header
    """
    response.headers["HX-Retarget"] = target
    return response


class HTMXResponse(HTMLResponse):
    """
    Enhanced HTMLResponse with HTMX helper methods.

    Usage:
        response = HTMXResponse(content="<div>Hello</div>")
        response.trigger("showNotification", {"message": "Success!"})
        return response
    """

    def trigger(
        self,
        event_name: str,
        detail: Optional[Dict[str, Any]] = None,
        after: str = "receive"
    ) -> "HTMXResponse":
        """Add trigger event to response."""
        htmx_trigger(self, event_name, detail, after)
        return self

    def refresh(self) -> "HTMXResponse":
        """Tell client to refresh page."""
        htmx_refresh(self)
        return self

    def reswap(self, swap_method: str) -> "HTMXResponse":
        """Override swap method."""
        htmx_reswap(self, swap_method)
        return self

    def retarget(self, target: str) -> "HTMXResponse":
        """Override target element."""
        htmx_retarget(self, target)
        return self


def get_htmx_headers(request: Request) -> Dict[str, str]:
    """
    Extract all HTMX-specific headers from a request.

    Useful for debugging and understanding HTMX request context.

    Args:
        request: FastAPI Request object

    Returns:
        Dict of HTMX headers
    """
    htmx_headers = {
        "hx_request": request.headers.get("HX-Request", ""),
        "hx_trigger": request.headers.get("HX-Trigger", ""),
        "hx_trigger_name": request.headers.get("HX-Trigger-Name", ""),
        "hx_target": request.headers.get("HX-Target", ""),
        "hx_current_url": request.headers.get("HX-Current-URL", ""),
        "hx_prompt": request.headers.get("HX-Prompt", ""),
    }

    return {k: v for k, v in htmx_headers.items() if v}


# Common swap methods for reference
SWAP_METHODS = {
    "innerHTML": "Replace the inner html of the target element",
    "outerHTML": "Replace the entire target element",
    "beforebegin": "Insert before the target element",
    "afterbegin": "Insert before the first child of the target",
    "beforeend": "Insert after the last child of the target",
    "afterend": "Insert after the target element",
    "delete": "Delete the target element",
    "none": "Do not swap content"
}
