"""Experimental cancellation ledger; a recorded goal never authorizes a write."""
import re

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
    return bool(ids) and all(i in quote for i in ids) if scope == 'selected' else not ids


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
