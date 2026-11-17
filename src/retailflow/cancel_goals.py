"""Experimental cancellation ledger; a recorded goal never authorizes a write."""
import re
from copy import deepcopy

GOAL_TOOL = {'type': 'function', 'function': {
    'name': 'update_cancel_goal',
    'description': 'Record the current requested cancellation scope. This does not cancel or authorize anything. Use none when cancellation is withdrawn or not requested.',
    'parameters': {'type': 'object', 'properties': {
        'scope': {'type': 'string', 'enum': ['none', 'all_pending', 'selected']},
        'order_ids': {'type': 'array', 'items': {'type': 'string'}},
        'source_quote': {'type': 'string'}},
        'required': ['scope', 'order_ids', 'source_quote'], 'additionalProperties': False}}}

GOAL_PROMPT = ('Read the customer messages in chronological order. Record the latest cancellation intent using one native '
               'update_cancel_goal call. Later corrections override earlier scope. Identity or reason answers do not cancel '
               'an earlier request. For all_pending or none use an empty order_ids list. For selected use only exact order IDs '
               'the customer supplied. source_quote must quote the customer message establishing the latest scope. Never output plain text.')

GOAL_REFERENCE_TOOL = deepcopy(GOAL_TOOL)
GOAL_REFERENCE_TOOL['function']['parameters'] = {
    'type': 'object', 'properties': {
        'scope': {'type': 'string', 'enum': ['none', 'all_pending', 'selected']},
        'order_indices': {'type': 'array', 'items': {'type': 'integer'}},
        'source_message_index': {'type': 'integer'}},
    'required': ['scope', 'order_indices', 'source_message_index'], 'additionalProperties': False}
GOAL_REFERENCE_PROMPT = (GOAL_PROMPT.replace('order_ids', 'order_indices')
    .replace('For selected use only exact order IDs the customer supplied.', 'For selected use zero-based indices from candidate_orders.')
    .replace('source_quote must quote', 'source_message_index must identify'))


def goal_reference_input(customer_messages):
    orders = list(dict.fromkeys(ident for text in customer_messages for ident in re.findall(r'#W\d+(?!\w)', text)))
    return {'customer_messages': [{'index': i, 'text': text} for i, text in enumerate(customer_messages)],
            'candidate_orders': [{'index': i, 'order_id': ident} for i, ident in enumerate(orders)]}


def resolve_goal_references(args, customer_messages):
    if set(args) != {'scope', 'order_indices', 'source_message_index'}:
        return None
    source, indices = args['source_message_index'], args['order_indices']
    orders = goal_reference_input(customer_messages)['candidate_orders']
    if type(source) is not int or not 0 <= source < len(customer_messages):
        return None
    if not isinstance(indices, list) or not all(type(i) is int and 0 <= i < len(orders) for i in indices):
        return None
    goal = {'scope': args['scope'], 'order_ids': [orders[i]['order_id'] for i in indices],
            'source_quote': customer_messages[source]}
    return goal if validate_goal(goal, customer_messages) else None


def validate_goal(args, customer_messages):
    if set(args) != {'scope', 'order_ids', 'source_quote'}:
        return False
    scope, ids, quote = args['scope'], args['order_ids'], args['source_quote']
    if scope not in ('none', 'all_pending', 'selected') or not isinstance(quote, str) or not quote or quote not in customer_messages:
        return False
    if not isinstance(ids, list) or not all(isinstance(i, str) and re.fullmatch(r'#W\d+', i) for i in ids):
        return False
    if len(set(ids)) != len(ids):
        return False
    if scope != 'none':
        if not re.search(r'\bcancel(?:led|ed|ling|ing|lation)?\b', quote, re.I):
            return False
        source = max(i for i, text in enumerate(customer_messages) if text == quote)
        kept = {ident for text in customer_messages[source:] for ident in re.findall(
            r'\b(?:keep|do not cancel|don\x27t cancel)\s+(?:order\s+)?(#W\d+)\b', text, re.I)}
        if kept and (scope == 'all_pending' or kept.intersection(ids)):
            return False
    return bool(ids) and all(re.search(re.escape(i) + r'(?!\w)', quote) for i in ids) if scope == 'selected' else not ids


def goal_progress(goal, user_id, business, inventory, committed):
    if goal.get('needs_refresh') or 'scope' not in goal:
        return {'complete': False, 'reason': 'goal_registration_required'}
    if goal['scope'] == 'none':
        return {'complete': True, 'orders': [], 'remaining_order_ids': []}
    if not user_id:
        return {'complete': False, 'reason': 'identity_required'}
    if goal.get('targets') is None:
        if goal['scope'] == 'all_pending':
            if inventory is None:
                return {'complete': False, 'reason': 'user_order_inventory_required'}
            goal['targets'] = [i for i in inventory if i in business['orders']
                               and business['orders'][i]['user_id'] == user_id
                               and business['orders'][i]['status'] == 'pending']
        else:
            goal['targets'] = list(goal['order_ids'])
    orders = []
    for ident in goal['targets']:
        order = business['orders'].get(ident)
        owned = order is not None and order['user_id'] == user_id
        orders.append({'order_id': ident, 'status': order['status'] if owned else 'unavailable',
                       'committed': bool(owned and order['status'] == 'cancelled' and ident in committed)})
    remaining = [o['order_id'] for o in orders if not o['committed']]
    return {'complete': not remaining, 'orders': orders, 'remaining_order_ids': remaining}
