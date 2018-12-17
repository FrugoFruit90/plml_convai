import sqlite3

import numpy as np

from nlp import normalize


# loading databases
domains = ['restaurant']#, 'hotel', 'attraction', 'train', 'taxi', 'hospital']#, 'police']
dbs = {}
for domain in domains:
    db = 'db/{}-dbase.db'.format(domain)
    conn = sqlite3.connect(db)
    c = conn.cursor()
    dbs[domain] = c


def oneHotVector(num_available_entities):
    """Return number of available entities for a particular domain.

    num_available_entitites: number of entities in the database
     that satisfy the current requirements of the user.
    """
    # TODO TASK 2
    # Create a one-hot encoding informing how many entities are available to the user
    available_entities = np.array([0, 0, 0, 0, 0, 0])
    # TODO

    return available_entities


def queryResult(domain, turn):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    sql_query = "select * from {}".format(domain)

    flag = True
    #print turn['metadata'][domain]['semi']
    for key, val in turn['metadata'][domain]['semi'].items():
        if key == 'requested' or val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                #val2 = normalize(val2)
                # change query for trains
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                #val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    #try:  # "select * from attraction  where name = 'queens college'"
    #print sql_query
    #print domain
    num_entities = len(dbs[domain].execute(sql_query).fetchall())

    return num_entities


def queryResultVenues(domain, turn, real_belief=False):
    # query the db
    sql_query = "select * from {}".format(domain)

    if real_belief == True:
        items = turn.items()
    else:
        items = turn['metadata'][domain]['semi'].items()

    flag = True
    for key, val in items:
        if key == "requested" or val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " +key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    try:  # "select * from attraction  where name = 'queens college'"
        return dbs[domain].execute(sql_query).fetchall()
    except:
        return []  # TODO test it
