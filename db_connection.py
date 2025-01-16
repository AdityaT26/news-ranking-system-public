import pymongo

def get_database_connection():

    connection_string = (
        #Put your connection string
    )
    client = pymongo.MongoClient(connection_string)
    db = client["unbiasly"]
    return db
