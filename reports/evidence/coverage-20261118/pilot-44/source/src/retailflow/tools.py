"""Retail tool schemas; actual business logic remains upstream."""

import ast
import math
import operator

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
    "exchange_delivered_order_items": {"order_id": str, "item_ids": list, "new_item_ids": list, "payment_method_id": str},
    "modify_pending_order_items": {"order_id": str, "item_ids": list, "new_item_ids": list, "payment_method_id": str},
    "calculate": {"expression": str},
    "transfer_to_human_agents": {"summary": str},
}
IDENTITY_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
WRITE_TOOLS = {"cancel_pending_order", "return_delivered_order_items", "exchange_delivered_order_items", "modify_pending_order_items"}


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


def calculate_amount(expression):
    """Bounded arithmetic only; no eval, exponentiation, names, or function calls."""
    if not isinstance(expression, str) or len(expression) > 128:
        raise ValueError("Expression exceeds calculation budget")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        if len(list(ast.walk(tree))) > 64:
            raise ValueError("Expression exceeds calculation budget")
        operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = node.value
            elif isinstance(node, ast.BinOp) and type(node.op) in operations:
                value = operations[type(node.op)](visit(node.left), visit(node.right))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            else:
                raise ValueError("Only numbers and + - * / are allowed")
            if not math.isfinite(value) or abs(value) > 1e12:
                raise ValueError("Calculation exceeds amount budget")
            return value
        return str(round(float(visit(tree.body)), 2))
    except (SyntaxError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        raise ValueError("Invalid arithmetic expression") from exc


def invoke(db, name: str, args: dict):
    if name == "calculate":
        return calculate_amount(args["expression"])
    result = getattr(RetailTools(db), name)(**args)
    return result.model_dump(mode="json") if isinstance(result, BaseModel) else result
