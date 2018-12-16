# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:42:59 2018

@author: micha
"""

import json

# Enter your keys/secrets as strings in the following fields
credentials = {}  
credentials['CONSUMER_KEY'] = "a"  
credentials['CONSUMER_SECRET'] = "b"
credentials['ACCESS_TOKEN'] = "c"
credentials['ACCESS_SECRET'] = "d"

# Save the credentials object to file
with open("twitter_credentials_test.json", "w") as file:  
    json.dump(credentials, file)