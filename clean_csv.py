import numpy as np


def cleaning_data (csv):
    df = csv.drop(columns=['test_group'])
    df = df.drop(columns=['session_id'])
    df = df.drop(columns=['clientId'])
    df['converted'] = df['converted'].replace(np.nan, 0)
    df['converted'] = df['converted'].replace("yes", 1) #we need to define it as 0 or 1 to calculate probability
    df['eventlist'] = df['eventstring'].str.strip('[]').str.replace('\'','').str.split(', ')
    df.eventlist = df.eventlist.apply(lambda x: [elem for elem in x if 'order' not in elem])
    df = df[df['date'].notna()]
    df = df.dropna()
    return df



def get_training_set(test_data, seq_len):

    data = test_data

    event_enum = ['add_to_cart', 'homepage_view', 'product_view', 'list_impression', 'remove_from_cart', 'basket_open', 'product_click', 'clear_basket', 'add_all_to_cart']

    def mapper(events):
        encoded = []
        for event in events:
            index = event_enum.index(event)
            encoded.append(index+1)
        return encoded


    data_encoded = mapper(data)
    print(data_encoded)

    zero_array = [0 for i in range(seq_len - len(data))]

    data_encoded = zero_array + data_encoded

    ## Convert the Categorically Encoded EventList to a fixed size Matrix for inputting to the models. This requires getting the value of the longest sequence to use as the length.
    ## Shorter sequences will have 0's padded on the left

    X = np.array(data_encoded)

    ## Split the Dataset into training and testing sets.
    X_test = X
    return X_test
