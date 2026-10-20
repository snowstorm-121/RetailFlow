"""The fixed eight-tool retail surface; actual business logic remains upstream."""

from pydantic import BaseModel

from retailflow.upstream.tools import RetailTools


SCHEMAS = {
    "find_user_id_by_email": {"email": str},
    "find_user_id_by_name_zip": {"first_name": str, "last_name": str, "zip": str},
    "get_user_details": {"user_id": str},
    "get_order_details": {"order_id": str},
    "get_product_details": {"product_id": str},
    "cancel_pending_order": {"order_id": str, "reason": str},
    "return_delivered_order_items": {"order_id": str, "item_ids": list, "payment_method_id": str},
    "transfer_to_human_agents": {"summary": str},
}
IDENTITY_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
WRITE_TOOLS = {"cancel_pending_order", "return_delivered_order_items"}


def valid_arguments(name: str, args: dict) -> bool:
    expected = SCHEMAS[name]
    if not isinstance(args, dict) or args.keys() != expected.keys():
        return False
    for key, kind in expected.items():
        value = args[key]
        if kind is list:
            if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
                return False
        elif not isinstance(value, kind) or not value:
            return False
    return True


def invoke(db, name: str, args: dict):
    result = getattr(RetailTools(db), name)(**args)
    return result.model_dump(mode="json") if isinstance(result, BaseModel) else result
