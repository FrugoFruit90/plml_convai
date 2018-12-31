import re

import simplejson as json

from .nlp import normalize

digitpat = re.compile('\d+')
timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat2 = re.compile("\d{1,3}[.]\d{1,2}")


def prepareSlotValuesIndependent():
    domains = ['restaurant']
    delex_list = []

    # TODO TASK 4
    # placeholders
    delex_area = []
    delex_food = []
    delex_price = []

    # read databases
    for domain in domains:
        fin = open('db/' + domain + '_db.json')
        db_json = json.load(fin)
        fin.close()

        for ent in db_json:
            for key, val in list(ent.items()):
                if val == '?' or val == 'free':
                    pass
                elif key == 'address':
                    delex_list.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    if "road" in val:
                        val = val.replace("road", "rd")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif "rd" in val:
                        val = val.replace("rd", "road")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif "st" in val:
                        val = val.replace("st", "street")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif "street" in val:
                        val = val.replace("street", "st")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                elif key == 'name':
                    delex_list.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    if "b & b" in val:
                        val = val.replace("b & b", "bed and breakfast")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif "bed and breakfast" in val:
                        val = val.replace("bed and breakfast", "b & b")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif "hotel" in val and 'gonville' not in val:
                        val = val.replace("hotel", "")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif "restaurant" in val:
                        val = val.replace("restaurant", "")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                elif key == 'postcode':
                    delex_list.append((normalize(val), '[' + domain + '_' + 'postcode' + ']'))
                elif key == 'phone':
                    delex_list.append((val, '[' + domain + '_' + 'phone' + ']'))

                # TODO TASK 4
                # Add delexicalized tokens for three slots key: area, food and pricerange
                # to the list of all possible dictionary slot-value pairs.
                # The tokens should have the form '[value_NAME_OF_THE_SLOT]'.

                elif key == 'area':
                    delex_area.append((normalize(val), '[' + 'value' + '_' + 'area' + ']'))
                elif key == 'food':
                    delex_food.append((normalize(val), '[' + 'value' + '_' + 'food' + ']'))
                elif key == 'pricerange':
                    delex_price.append((normalize(val), '[' + 'value' + '_' + 'pricerange' + ']'))
                else:
                    pass

    # more general values add at the end
    delex_list.extend(delex_area)
    delex_list.extend(delex_food)
    delex_list.extend(delex_price)

    return delex_list


def delexicalise(utt, dictionary):
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?

    return utt


def delexicaliseDomain(utt, dictionary, domain):
    for key, val in dictionary:
        if key == domain or key == 'value':
            utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
            utt = utt[1:-1]  # why this?

    # go through rest of domain in case we are missing something out?
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?
    return utt

if __name__ == '__main__':
    prepareSlotValuesIndependent()
