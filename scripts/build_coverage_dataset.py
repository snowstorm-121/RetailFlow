"""Build the reviewed broader-task version; source identities/answers never enter openings."""
import json
from pathlib import Path
from retailflow.dataset import ROOT, prepare_dataset
from retailflow.upstream.paths import RETAIL_DB_PATH

DB = json.loads(RETAIL_DB_PATH.read_text())
SOURCES = {t['id']: t for t in json.loads((RETAIL_DB_PATH.parent / 'tasks.json').read_text())}
CFG = json.loads((ROOT / 'configs/exchange-evaluation.json').read_text())
M='modify_pending_order_items'; E='exchange_delivered_order_items'; R='return_delivered_order_items'; C='cancel_pending_order'
ADAPTERS=[]


def variant(name, new, old=None):
    return {'name':name,'new_options':new,'old_options':old or {}}


def consent(tool, requirements, payment=None):
    c={'tool':tool,'include':list(dict.fromkeys(r['name'] for r in requirements)), 'exchange_requirements':requirements}
    if payment:c['exchange_payment_source']=payment
    return c


def add(id, opening, user_consent, replies=None, rules=None, note='', review=None):
    source=SOURCES[id]; actions=source['evaluation_criteria'].get('actions') or []
    order_ids=[a['arguments']['order_id'] for a in actions if 'order_id' in a['arguments']]
    if order_ids:uid=DB['orders'][order_ids[0]]['user_id']
    else:uid=next(a['arguments']['user_id'] for a in actions if a['name']=='get_user_details')
    # Identity is supplied only when the agent requests it. Preserve source false-email probes.
    known=source['user_scenario']['instructions']['known_info']
    profile=DB['users'][uid]
    if profile['name']['first_name'] not in known and profile['email'] not in known:
        known=f"My name is {profile['name']['first_name']} {profile['name']['last_name']}, ZIP {profile['address']['zip']}."
    fields={'identity':known,'order':"I don't remember; please look through my orders.",
            'payment_method':'The original payment method.', 'refund_method':'The original payment method.',
            'exchange_items':opening,'exchange_options':opening,'return_items':opening, **(replies or {})}
    row={'id':id,'category':'multiturn_change' if rules else 'legal_write','user_id':uid,
         'opening':opening,'field_responses':fields}
    if rules:
        row['proposal_rules']=rules;row['required_proposal_rules']=list(range(len(rules)))
    CFG['selected_tasks'].append(row);CFG['user_consent']['tau-retail-'+id]=user_consent
    ADAPTERS.append({'id':id,'user_id':uid,'adaptation':note or 'Full business intent retained; emotional style not simulated. Unspecified payment defaults to original method only when asked.',
                     'semantic_review':review or ['Check every original user subgoal and all communication, not only reference writes.','No unsupported settlement, email or shipment claims.']})

shirt={'color':'purple','size':'S','material':'polyester','style':'v-neck'}
add('3','How many T-shirt variants are available? Modify all my pending small T-shirts to purple, keeping size S and v-neck, preferably polyester.',consent(M,[variant('T-Shirt',shirt,{'size':'S'})]))
add('4','How many T-shirt variants are available? Modify all my pending T-shirts to purple size S, v-neck, preferably polyester.',consent(M,[variant('T-Shirt',shirt)]),note='Interprets source requested v-neck as target style for all shirts; source calls it same v-neck although one original is crew neck. This explicit local adaptation is not an unchanged official task.')
for id,brightness,power in [('6','low','battery'),('7','low','AC adapter'),('9','high','AC adapter')]:
    direction='less bright' if brightness=='low' else 'brighter'
    preference='battery, then USB, then AC' if power=='battery' else 'AC adapter, then battery, then USB'
    add(id,f'Exchange my water bottle for a bigger one, and my desk lamp for a {direction} one. For the lamp I prefer {preference}.',
        consent(E,[variant('Desk Lamp',{'brightness':brightness,'power source':power})]),
        rules=[{'tool':E,'decision':'revise','reply':f'Actually, exchange only the desk lamp. Keep the bottle. I still want it {direction}, preferring {preference}.'}],
        note='Confirmation-triggered revision is kept out of opening and required by the scorer.')
add('5','Exchange my water bottle for a bigger one and my desk lamp for a less bright one, preferring battery, then USB, then AC.',
    {'tool':R,'include':['Water Bottle']},
    rules=[{'tool':E,'decision':'revise','reply':'Actually, only exchange the desk lamp; keep the water bottle.'},
           {'tool':E,'decision':'revise','reply':'Do not exchange anything. Return only the water bottle to the original payment method instead.'}],
    note='Both confirmation-dependent revisions must occur; rejected proposals cannot change business state.')
add('12','I got into gaming and want to cancel or return everything unrelated to gaming. I want PayPal for the refund; if you cannot do that, hand this to a human.',{},replies={'refund_method':'PayPal only. Transfer to a human if unavailable.','payment_method':'PayPal only. Transfer to a human if unavailable.','return_items':'Everything except the keyboard and mouse.'},note='Gaming subset remains private; explicit conditional escalation replaces anger. Must explain refund restriction, leave orders intact and record local handoff.')
add('18','My office chair arrived with broken pieces. I want to return it.',
    consent(E,[variant('Office Chair',{'material':'leather','color':'gray','armrest':'fixed','backrest height':'high-back'})]),
    rules=[{'tool':R,'decision':'revise','reply':'I changed my mind. Exchange for the same chair if available. Otherwise, get a gray leather chair with fixed armrests.'}],
    note='Retains return-to-exchange revision; unavailable-item fallback is explicitly in the revision reply rather than a separate refusal-triggered turn.')
add('19','Return my water bottle and exchange my pet bed and office chair to their cheapest versions. I want both; if only one option is possible, choose whichever saves more money and tell me the savings for both.',{'tool':R,'include':['Water Bottle']},replies={'return_items':'Only the water bottle.'},review=['Explain the once-per-order return/exchange conflict.','Compare $54.04 return to $41.64 exchange saving, and choose return.'])
add('20','Upgrade all my items to the most expensive available variants. Keep the shoe size unchanged; other features do not matter. If an order cannot be changed, explain and proceed to the next order. Use my gift card, or PayPal if that is impossible.',
    consent(M,[variant('Water Bottle',{'capacity':'750ml','material':'glass','color':'black'}),variant('Mechanical Keyboard',{'switch type':'linear','backlight':'RGB','size':'full size'}),variant('Running Shoes',{'size':'9','color':'black','material':'synthetic','sole':'rubber'}),variant('Makeup Kit',{'skin tone':'medium','kit size':'professional','brand':'Brand A'})],'gift_card'),
    replies={'payment_method':'Use my gift card; PayPal only if that cannot cover it.'},note='Preferences resolved against fixed available catalogue: most expensive variants, shoe size9 retained; $71.96 difference fits $86 gift balance. Other orders processed, must be explained unchanged.')
add('23','Exchange my helmet to medium, red, high ventilation; exchange my luggage set in another order to a black two-piece softshell set. Also modify my newly ordered grill to the same type as the grill I already received.',
    {'by_tool':{E:{'by_order':{'#W3947049':consent(E,[variant('Cycling Helmet',{'size':'M','color':'red','ventilation':'high'})]),'#W6876713':consent(E,[variant('Luggage Set',{'piece count':'2-piece','color':'black','material':'softshell'})])}},M:consent(M,[variant('Grill',{'type':'charcoal','size':'medium','features':'rotisserie'})])}})
add('27','Return my garden hose and backpack, and exchange my hiking boots for exactly the same specs except waterproof. If the order permits only one operation, I prefer the exchange.',consent(E,[variant('Hiking Boots',{'size':'12','material':'leather','waterproof':'yes'})]),review=['Explain return/exchange conflict and leave hose/backpack unreturned.','Confirm waterproof boots with same size/material.'])
add('29','Exchange my skateboard for a shorter bamboo one. Tell me all available options and prices; choose the most expensive. Also exchange my delivered garden hose to the type in my pending order. Do not cancel anything.',
    {'tool':E,'by_order':{'#W3792453':consent(E,[variant('Skateboard',{'length':'28 inch','deck material':'bamboo','design':'plain'})]),'#W7181492':consent(E,[variant('Garden Hose',{'length':'50ft','material':'vinyl','color':'black'})])}},
    note='Private hose specification is resolved from the pending order, never exposed as a target item ID. Wrong proposals are rejected; correction reply points back to pending order.',review=['List shorter bamboo options $180.10/$189.57/$208.60 and select highest.','Match hose to pending order; no cancellation.'])
add('35','Return the more expensive speaker that is not water resistant. Modify the 17-inch laptop in another order to 13-inch. If the exact version is unavailable, list all available 13-inch options; I prefer i5 over i7 and silver or black over other colors.',
    {'by_tool':{R:{'include':['Bluetooth Speaker'],'return_options':{'Bluetooth Speaker':{'water resistance':'no'}}},M:consent(M,[variant('Laptop',{'screen size':'13-inch','processor':'i5','ram':'16GB','storage':'1TB SSD','color':'silver'},{'screen size':'17-inch'})])}},replies={'return_items':'The more expensive speaker that is not water resistant.'})
add('36','My recent order exceeds $1160 but my card has only $1131 credit. Can you split payment? If not, tell me the most expensive item and price and whether you can cancel only it. If not, change all items to their cheapest variants if the total is at most $1131; otherwise cancel the order.',
    consent(M,[variant('Action Camera',{'resolution':'4K','waterproof':'yes','color':'black'}),variant('Desk Lamp',{'color':'black','brightness':'medium','power source':'AC adapter'}),variant('T-Shirt',{'color':'red','size':'L','material':'cotton','style':'v-neck'})]),replies={'cancel_reason':'I no longer need it.'},review=['Explain no split payment or single-item cancellation.','Identify camera $481.50 and cheapest full order $1130.85 under $1131.'])
add('38','My recent order exceeds $1100 but my card has only $950 credit. Can you split payment? If not, tell me the most expensive item and price and whether you can cancel only it. If not, check whether cheapest variants bring the whole order to at most $950. If they do not, cancel the order so I can reorder.',{'tool':C,'all':True,'cancel_reason':'no longer needed'},replies={'cancel_reason':'I no longer need it.'},review=['Explain split/single-item limits, camera $481.50, minimum order $1130.85 exceeds $950, then cancel.'])
add('44','Change the desk lamp in #W9300146 to the cheapest available desk lamp. Put any refund onto my gift card and tell me how much I get back.',consent(M,[variant('Desk Lamp',{'color':'black','brightness':'medium','power source':'AC adapter'})],'gift_card'),replies={'payment_method':'My gift card.'})
add('49','I mistakenly bought IPX7 wireless earbuds. Exchange those for the cheapest earbud variant among the other earbuds in that same order, matching that pair’s water resistance.',consent(E,[variant('Wireless Earbuds',{'color':'blue','battery life':'6 hours','water resistance':'IPX4'},{'water resistance':'IPX7'})]),note='Source final clarification cheapest among the other purchased earbuds is included up front; no target item IDs exposed.')
add('70','Exchange the helmet I received to medium and high ventilation. Prefer blue if available. Use the original payment method and tell me the price difference due.',consent(E,[variant('Cycling Helmet',{'size':'M','color':'blue','ventilation':'high'})]))
add('94','Exchange a laptop I received to i7, 8GB RAM and 1TB SSD.',consent(E,[variant('Laptop',{'processor':'i7','ram':'8GB','storage':'1TB SSD'},{'screen size':'15-inch','ram':'16GB'})]),replies={'exchange_items':'The 15-inch laptop with 16GB RAM.'},note='Which-laptop information stays in private reply/consent. Same source user as93 stays in same local split.')
add('95','Exchange the laptop I received to i7, 8GB RAM and 1TB SSD. Tell me the individual differences and total amount due.',consent(E,[variant('Laptop',{'processor':'i7','ram':'8GB','storage':'1TB SSD'},{'screen size':'15-inch'})]),replies={'exchange_items':'Actually both 15-inch laptops, in the two orders.'},review=['Discover both15-inch laptops using private reply.','State $167.87 due, $60.78 refund due, net $107.09; not settled today in local environment.'])
add('100','Return my luggage set for the exact same type in red, and my skateboard in the same order for 34-inch custom. If these are still pending, modify that order instead, using my credit card. Also return my hiking boots.',
    {'by_tool':{M:consent(M,[variant('Luggage Set',{'piece count':'2-piece','color':'red','material':'softshell'}),variant('Skateboard',{'length':'34 inch','design':'custom'})],'credit_card'),R:{'include':['Hiking Boots']}}},replies={'payment_method':'My credit card for the pending modification; original method for the boots return.','return_items':'Only the hiking boots.'},note='Pending-modification fallback included in opening rather than waiting for rejection; full outcome preserved.')
add('105','For #W4316152, exchange one glass 2-liter induction tea kettle for ceramic/gas, and the other identical kettle for 1.5-liter/gas.',
    consent(E,[variant('Tea Kettle',{'material':'ceramic','stovetop compatibility':'gas'},{'material':'glass','capacity':'2 liters','stovetop compatibility':'induction'}),variant('Tea Kettle',{'capacity':'1.5 liters','stovetop compatibility':'gas'},{'material':'glass','capacity':'2 liters','stovetop compatibility':'induction'})]),note='Duplicate old item quantity and distinct replacement requirements are matched as a multiset, not collapsed by product name.')
add('66','Change my luggage set to a coat. If that cannot be done, return that item; if that is also impossible, cancel the whole order.',{'tool':C,'include':['Luggage Set'],'cancel_reason':'no longer needed'},replies={'cancel_reason':'I no longer need it.'},review=['Explain cross-product modification forbidden and pending item cannot be returned; cancel whole eligible order.'])

# Additional complete conditional goals using the same guarded tools.
add('16','Cancel all pending orders because I no longer need them. Return only the smart watch I received, nothing else. Tell me the total amount I can get back.',
    {'by_tool':{C:{'all':True,'cancel_reason':'no longer needed'},R:{'include':['Smart Watch']}}},
    replies={'cancel_reason':'I no longer need them.','return_items':'Only the smart watch.'},review=['Cancel both pending orders, return only delivered watch, explain total $8276.23 without treating requested return as settled.'])
add('30','My tablet arrived damaged. First tell me its tracking number. Exchange for the same item if possible; if unavailable, return it and cancel the charger I just bought for it. Finally return my sneakers.',
    {'by_tool':{R:{'include':['Tablet','Sneakers']},C:{'include':['Portable Charger'],'cancel_reason':'no longer needed'}}},
    replies={'cancel_reason':'I no longer need the charger.','return_items':'The tablet and sneakers.'},review=['Give tracking746342064230 before handling tablet.','Check catalogue unavailable tablet; return tablet, cancel charger, return sneakers.'])
for id,extra,includes in [('31','Cancel only the boots and keep the kettle; if partial cancellation is impossible, leave that order alone.',['Portable Charger']),('32','Also cancel both the boots and kettle.',['Portable Charger','Hiking Boots','Electric Kettle'])]:
    add(id,'I lost the tablet I received. Tell me the tracking number and whether you can refund or reorder it. If not, cancel the charger bought for it. '+extra+' Finally return my sneakers.',
        {'by_tool':{C:{'include':includes,'cancel_reason':'no longer needed'},R:{'include':['Sneakers']}}},
        replies={'cancel_reason':'I no longer need those orders.','return_items':'Only the sneakers.'},review=['Give tracking746342064230; explain lost item cannot be returned/reordered here.','Apply conditional cancellations exactly; return only sneakers.'])
add('54','I have financial problems. Cancel or return all eligible orders, except keep my beloved boots. You may exchange boots for the cheapest cheaper pair of exactly the same size and material. If none is cheaper, leave the boots alone and handle the other items. Tell me the total refund amount.',
    {'by_tool':{C:{'all':True,'cancel_reason':'no longer needed'},R:{'all':True,'exclude':['Hiking Boots']}}},
    replies={'cancel_reason':'I no longer need the other orders.','return_items':'Everything eligible except my hiking boots.'},review=['Verify no cheaper size12 leather boots; preserve boots order.','Cancel two pending orders and return all items in other delivered order; explain total3646.68 and unsettled return.'])
add('76','Remove the fleece jacket I ordered by mistake; if removing one item is impossible, cancel that whole order as ordered by mistake. Modify my skateboard to maple, 34 inch, graphic; if unavailable, cancel that order as no longer needed. Also tell me the total paid for all grills in prior orders.',
    {'tool':C,'by_order':{'#W8367380':{'include':['Fleece Jacket'],'cancel_reason':'ordered by mistake'},'#W1242543':{'include':['Skateboard'],'cancel_reason':'no longer needed'}}},
    replies={'cancel_reason':'Jacket order: ordered by mistake. Skateboard order: no longer needed.'},review=['Explain no single-item removal and unavailable target skateboard, cancel with distinct reasons.','Report total grill price1939.05.'])
# Reject conflicting source references rather than silently modifying initial balances or stock.
excluded={'18':'Fallback asks unavailable same chair, but original variant is in stock; policy prohibition is not stock unavailability.',
          '105':'Reference exchange needs $21.10, sole gift card balance is $17; no valid payment method in fixed state.'}
CFG['selected_tasks']=[r for r in CFG['selected_tasks'] if r['id'] not in excluded]
ADAPTERS=[r for r in ADAPTERS if r['id'] not in excluded]
for id in excluded:CFG['user_consent'].pop('tau-retail-'+id,None)

# Keep every previously assigned account in its existing split. New accounts use source train if any selected member is train.
old_manifest=json.loads((ROOT/'data/versions/t3-exchange-v1/data/manifests/dataset.json').read_text())
assignments={group.removeprefix('source-user:'):split for split,groups in old_manifest['groups'].items() for group in groups}
splits=json.loads((RETAIL_DB_PATH.parent/'split_tasks.json').read_text())
for row in CFG['selected_tasks']:
    uid=row['user_id']
    if uid not in assignments:
        members=[r['id'] for r in CFG['selected_tasks'] if r['user_id']==uid]
        assignments[uid]='dev' if any(id in splits['train'] for id in members) else 'test'
CFG.update(dataset_version='t4-coverage-v2', output_root='data/versions/t4-coverage-v2',dataset_root='data/versions/t4-coverage-v2',
           additional_tools=[E,M,'calculate'],group_assignments=assignments,experiment_version='coverage-20261118',
           revision_note='Broader local derived tasks with grouped local split preserving source_split. Upstream local item mutation patch disclosed. No claim of80 tasks, unseen Test or model success for reference checks.')
# Broad script permits correction without ever revealing gold labels to agent; specific replies retained above.
CFG['user_consent']['tau-retail-29']['rejection_reply']='That is not the requested variant. For the hose, look again at my pending order; do not cancel it.'
(ROOT/'configs/coverage-evaluation.json').write_text(json.dumps(CFG,ensure_ascii=False,indent=2)+'\n')
(ROOT/'data/reviews/coverage-adapters-20261118.json').write_text(json.dumps({'version':'coverage-20261118','adapters':ADAPTERS,'excluded_source_conflicts':excluded,'excluded_duplicate':{'37':'Same final goal as36; different thresholds both select identical modifications.'}},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
    print(json.dumps(prepare_dataset(CFG),ensure_ascii=False,indent=2))
