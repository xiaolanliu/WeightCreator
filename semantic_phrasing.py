"""Explicit visual noun phrases; authored language priors, not fitted classifiers.

These phrases express ordinary package/functional categories. They contain no
task IDs, object instances, scene locations, colors or pixel templates.
"""


def visual_phrase(noun):
    text=noun.lower().strip()
    text=text.removeprefix('the ')
    packages={'alphabet soup':'can', 'tomato sauce':'can',
              'cream cheese':'box', 'cream cheese box':'box',
              'salad dressing':'bottle', 'bbq sauce':'bottle', 'barbecue sauce':'bottle',
              'ketchup':'bottle', 'milk':'carton', 'orange juice':'carton',
              'chocolate pudding':'box', 'butter':'package'}
    if text in packages:
        product=text.removesuffix(' box')
        return f'a {packages[text]} of {product}'
    if text=='stove':return 'an electric stove burner'
    if text=='cookie box':return 'a box of cookies'
    if text=='caddy':return 'an open divided storage container'
    if text=='rack':return 'a rack with parallel slats'
    if text in ('cabinet','wooden cabinet'):return 'a cabinet with drawers'
    return 'an '+text if text[:1] in 'aeiou' else 'a '+text
