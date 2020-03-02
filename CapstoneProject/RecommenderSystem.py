#Recommender System is being modelled for Germany region
#The application models popular based, content based, item and user collaborative filterring and hybrid 

import pickle
from flask import Flask, request, jsonify, render_template
import json
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS, cross_origin

#Data Cleanup routine
#Fill missing Customer ID with default ID 99999 and the missing description as 'Desc Missing'
#Convert StockCode data as str type, StockCode is alphanumeric
#Convert CustomerID to integer
#StockCodes starting with 'C' are return transactions, we will filter them out as we are modelling only sale transactions. Also, trim spaces
#We will drop StockCode 'POST' them dont seem to add any insights into the model
def dataCleaning(germany_retail_df):
    germany_retail_df['CustomerID'].fillna(99999.0, inplace=True)
    germany_retail_df['Description'].fillna('Desc Missing', inplace=True)
    germany_retail_df['StockCode'] = germany_retail_df['StockCode'].astype(str)
    germany_retail_df['CustomerID']=germany_retail_df['CustomerID'].apply(np.int64) 
    germany_retail_df.StockCode = germany_retail_df.StockCode.str.replace(' ', '')
    german_cancelled_trans_df = germany_retail_df[germany_retail_df['InvoiceNo'].astype('str').str.startswith('C', na=False)]
    germany_retail_df = germany_retail_df.drop(germany_retail_df[germany_retail_df['StockCode'] == 'POST'].index, axis=0)
    return germany_retail_df

#The routine returns the Sale transactions in Germany region
def getSaleData(german_dataset_df):
    #Remove Cancelled transcations from the dataset
    german_cancelled_trans_df = german_dataset_df[german_dataset_df['InvoiceNo'].astype('str').str.startswith('C', na=False)]
    german_sale_trans_df = german_dataset_df[~german_dataset_df['InvoiceNo'].isin(german_cancelled_trans_df.InvoiceNo)]
    return german_sale_trans_df

#Pre-process the data
#Filter large quantity data [products having quantity > 299) and replace when with the mean of the quantity
#Drop columns 'InvoiceNo','Country', 'InvoiceDate', 'Description', 'UnitPrice' as they are not needed for the model.
#Reposition CustomerID column to the start of the dataframe
def dataPreprocessing(german_sale_trans_df):
    german_sale_trans_df['Quantity']=np.where(german_sale_trans_df['Quantity'] > 299, german_sale_trans_df.Quantity.mean(), german_sale_trans_df['Quantity'])
    german_sale_trans_df.drop(labels=['InvoiceNo','Country', 'InvoiceDate', 'Description', 'UnitPrice'], axis=1, inplace=True)
    front = german_sale_trans_df['CustomerID']
    german_sale_trans_df.drop(labels=['CustomerID'], axis=1,inplace = True)
    german_sale_trans_df.insert(0, 'CustomerID', front)
    return german_sale_trans_df

#Routine which maps the StockCode to the description of the product. If a product has multiple descriptions, the first description is considerred.
#Drop duplicate description names
def createStockDescMapperDF(german_sale_trans_df):
    german_stock_desc_df = german_sale_trans_df[['StockCode', 'Description']]
    german_stock_desc_df.reset_index(inplace=True)
    german_stock_desc_df.drop(['index'], axis=1, inplace=True)
    test=german_stock_desc_df.groupby(['StockCode','Description']).count()
    german_stock_desc_df = test.reset_index()
    german_stock_desc_df.drop_duplicates(subset='StockCode', keep="first", inplace=True)
    german_stock_desc_df.reset_index(inplace=True)
    german_stock_desc_df.drop('index',axis=1, inplace=True)
    return german_stock_desc_df

#Routine which returns a dataframe of the quantity of a stock purchased by a customer. Higher the qty of a product purchased indicates the inclination of the cusomter towards the product
def createCustomerStockQtyMapperDF(german_sale_trans_df):
    german_sale_trans_group_df = german_sale_trans_df.groupby(['CustomerID', 'StockCode']).sum().sort_values(by='Quantity', ascending=False)[['Quantity']]
    german_sale_trans_group_df.reset_index(inplace=True)
    return german_sale_trans_group_df

#Helper Routines
#printProducts - Return the stockCode and the desription of the product for the products in the productList
#getStockCodeIndex - Returns the index of stockCode from the pivot table of items
#getCustomerIDIndex - Returns the index of stockCode from the pivot table of customers
#productsToRecommend - Returns the top 10 products from the recommended list
#similarCustomers - Return the customers who are similar based on the distance
#filter_top10 - routine which compares two lists of recommendations and returns the common items in the two plus the items from the first list
#make_recommendation_collab - routine which returns the nearest neighbours based on item similarity or user similarity depending on input
#fallbackPopularItems - Returns the popular items in offline (new customer or new item) and exception scenarios
def printProducts(productList):
    descList =[]
    print('Products to Recommend: ',productList)
    return german_stock_desc_df[german_stock_desc_df['StockCode'].isin(productList)]

def getStockCodeIndex(productCode):
    try:
        stockCodeIndex = temp_list.index(productCode)
        return stockCodeIndex
    except:
        return -1;

def getCustomerIDIndex(custID):
    try:
        custIDIndex = temp_user_list.index(custID)
        return custIDIndex
    except:
        return -1;

def productsToRecommend(recommends):
    product_list = []
    threshold = 10
    for i in range(threshold):
            product_list.append(temp_list[recommends[i][0]])
    return product_list

def similarCustomers(recommends):
    customer_list = []
    distance_list = []
    threshold = 10
    for i in range(threshold):
            customer_list.append(temp_user_list[recommends[i][0]])
            distance_list.append(recommends[i][1])
    
    customer_dist_dict = dict(zip(customer_list, distance_list))
    return customer_dist_dict

def filter_top10(top10_collab, top10_content):
    primary_matches = list(set(top10_collab).intersection(set(top10_content)))
    print('Common stockCode: \n', primary_matches, '\n')
    secondary_matches = [ele for ele in top10_collab if ele not in primary_matches]
    return primary_matches + secondary_matches

def make_recommendation_collab(model_knn, data, index_value, n_recommendations):
    distances, indices = model_knn.kneighbors(data[index_value], n_neighbors=n_recommendations+1)
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    return raw_recommends

#Popular based - returns the top 10 products sold in Germany based on the quantity
def popularGermanItems():
    purchase_g = german_sale_trans_df_orig.groupby(['StockCode'])['Quantity'].sum().reset_index()
    purchase_g_sorted = purchase_g.sort_values("Quantity", ascending=False).reset_index().head(10)
    top_selling_stockcodes = purchase_g_sorted['StockCode'].head(10).tolist()
    return top_selling_stockcodes

#Populating popularItems during cold start (new item/new user) or exception scenario
def fallbackPopularItems():
    print('No products to recommend','\n')
    print('Take a look at popular products.....', '\n')
    prod_desc = "No Matching items - Below are the Popular items sold"
    dictList = printProducts(popularGermanItems())
    res = dictList.to_dict('records')
    res.append(prod_desc)
    return res

#Content based filterring    
#Function that takes in stockCode as the input, maps it to the product description and outputs most similar products based on the description
#Recomendation is based using Term Frequency of the words in the description and using cosine similarity
def get_recommendations_content(stockCode):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(german_stock_desc_df['Description'])
        doc_term_matrix = tfidf_matrix.todense()
        df = pd.DataFrame(doc_term_matrix, 
                  columns=tfidf.get_feature_names(), 
                  index=german_stock_desc_df['Description'])
        from sklearn.metrics.pairwise import linear_kernel
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        from sklearn.metrics.pairwise import cosine_similarity
        indices = pd.Series(german_stock_desc_df.index, index=german_stock_desc_df['Description'])
    
        desc = german_stock_desc_df[german_stock_desc_df['StockCode'] == stockCode]['Description'].tolist()[0]
        idx = indices[desc]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        product_indices = [i[0] for i in sim_scores]
        content_stocks_df = german_stock_desc_df[german_stock_desc_df['StockCode'].isin(german_stock_desc_df['StockCode'].iloc[product_indices])]
        top10product_content_list = content_stocks_df['StockCode'].tolist()
        return content_stocks_df
    except:
        return fallbackPopularItems()

#Item based collaborative filterring
#Uses the customer and item feature matrix to get the items that are similar based on the quantity of purchase by the customer
def itemBasedCollabFilter(productCode):
    model_knn.fit(cust_item_features)
    try:
        stockIndex = getStockCodeIndex(productCode)
        prod_desc_tup = productCode, ' - ', german_sale_trans_df_orig[german_sale_trans_df_orig['StockCode'] == productCode][['Description']][-1:]['Description'].iloc[0]
        prod_desc = ''.join(prod_desc_tup)
        if(stockIndex == -1):
            print('StockCode not Recognised!!!','\n')
            print('Take a look at popular products.....','\n')
            prod_desc = "StockCode not found! Below are the Popular items sold"
            dictList = printProducts(popularGermanItems())
            res = dictList.to_dict('records')
            res.append(prod_desc)
            return res
        else:
            recommends = make_recommendation_collab(
            model_knn=model_knn,
            data=cust_item_features,
            index_value=stockIndex,
            n_recommendations=10)
    
            print('\n Product Recomendations for ',productCode, '-', german_sale_trans_df_orig[german_sale_trans_df_orig['StockCode'] == productCode][['Description']][-1:]['Description'].iloc[0])
            top10product_collab_list = productsToRecommend(recommends)
            if(len(top10product_collab_list) >= 1):
                print('Top 10 StockCodes - Item Collab \n',top10product_collab_list)
                dictList = printProducts(top10product_collab_list)
                res=dictList.to_dict('records')
                res.append(prod_desc)
                return res
            else:
                return fallbackPopularItems()
    except:
        return fallbackPopularItems()
        
#User based
#Uses the customer and item feature matrix to get the users that are similar based on the quantity of purchase of the items
def userBasedCollabFilter(customerID):
    model_knn.fit(cust_user_features)
    try:
        customerIDIndex = getCustomerIDIndex(customerID)
        if(customerIDIndex == -1):
            print('Customer ID not Recognised!!!','\n')
            print('Take a look at popular products.....','\n')
            prod_desc = "CustomerID not found! Below are the Popular items sold"
            dictList = printProducts(popularGermanItems())
            res = dictList.to_dict('records')
            res.append(prod_desc)
            return res
        else:
            recommends = make_recommendation_collab(
            model_knn=model_knn,
            data=cust_user_features,
            index_value=customerIDIndex,
            n_recommendations=10)
    
            print('\n Product Recomendations for Customer - ',customerID)
            similarCustomersDict = similarCustomers(recommends)
            print(similarCustomersDict)
            print('\n')
            dictEmpty = bool(similarCustomersDict)
            if(dictEmpty):
                most_similar_customer= list(similarCustomersDict.keys())[0]
                #Get top 10 items for this customer and we will recommend these to our target customer
        
                #Get the list of items (top 10) not purchased by target customer but most frequently purchased by similar customer 
                target_customer_item_list = german_sale_trans_group_df[german_sale_trans_group_df['CustomerID'] == customerID]['StockCode'].tolist()
                most_similar_cust_item_list = german_sale_trans_group_df[german_sale_trans_group_df['CustomerID'] == most_similar_customer]['StockCode'].tolist()

                items_not_purchased = [item for item in most_similar_cust_item_list if item not in target_customer_item_list] 
        
                cust_stock_item_df = german_sale_trans_group_df[(german_sale_trans_group_df['CustomerID'] == most_similar_customer) & (german_sale_trans_group_df['StockCode'].isin(items_not_purchased))].sort_values(by='Quantity', ascending=False)
                qty_mean = cust_stock_item_df['Quantity'].mean()
        
                #Get top 10 stockCodes sold more than the mean qty
                first_cust_top10_stockCodes = cust_stock_item_df[cust_stock_item_df['Quantity'] >= qty_mean]['StockCode'].head(10)#.to_string(index=False)
                print('1st Customer - Top 10 StockCodes')
                print(first_cust_top10_stockCodes.values)
                print('\n')
                #Check the next closest customer to the target customer and evaulated the item basket if the distance > .75
                try:
                    if(list(similarCustomersDict)[1]):
                        second_closest_customer = list(similarCustomersDict)[1]    
                        distance = similarCustomersDict[second_closest_customer]
                        if(distance > 0.75):
                            print('2nd Customer - Top 10 StockCodes')
                            second_most_similar_cust_item_list = german_sale_trans_group_df[german_sale_trans_group_df['CustomerID'] == second_closest_customer]['StockCode'].tolist()
                            second_cust_items_not_purchased = [item for item in second_most_similar_cust_item_list if item not in target_customer_item_list] 
                            second_cust_stock_item_df = german_sale_trans_group_df[(german_sale_trans_group_df['CustomerID'] == second_closest_customer) & (german_sale_trans_group_df['StockCode'].isin(second_cust_items_not_purchased))].sort_values(by='Quantity', ascending=False)
                            second_cust_qty_mean = second_cust_stock_item_df['Quantity'].mean()
                    
                            second_cust_top10_stockCodes = second_cust_stock_item_df[second_cust_stock_item_df['Quantity'] > second_cust_qty_mean]['StockCode'].head(10)#.to_string(index=False)
                            print(second_cust_top10_stockCodes.values)
                            print('\n')
                            prod_to_recommend = filter_top10(first_cust_top10_stockCodes, second_cust_top10_stockCodes)
                            if(len(prod_to_recommend) >= 1):
                                cust_desc_tup = "Product Recomendations for Customer - ",str(customerID)
                                cust_desc = ''.join(cust_desc_tup)
                                dictList = printProducts(prod_to_recommend)
                                res = dictList.to_dict('records')
                                res.append(cust_desc)
                                return res
                            else:
                                return fallbackPopularItems()
                except IndexError:
                    print('Invalid')
            else:
                return fallbackPopularItems()
    except:
        return fallbackPopularItems()

#Hybrid based filterring
#Uses Item based collaborative filterring plus content based filterring to recommend items that are common and the items purchased by the nearest neighbour
def hybridBasedModel(productCode):
    model_knn.fit(cust_item_features)
    try:
        stockIndex = getStockCodeIndex(productCode)
        if(stockIndex == -1):
            print('StockCode not Recognised!!!','\n')
            print('Take a look at popular products.....','\n')
            prod_desc = "StockCode not found! Below are the Popular items sold"
            dictList = printProducts(popularGermanItems())
            res = dictList.to_dict('records')
            res.append(prod_desc)
            return res
        else:
            recommends = make_recommendation_collab(
            model_knn=model_knn,
            data=cust_item_features,
            index_value=stockIndex,
            n_recommendations=10)
    
            top10product_collab_list = productsToRecommend(recommends)
            print('\n Product Recomendations for ',productCode, '-', german_sale_trans_df_orig[german_sale_trans_df_orig['StockCode'] == productCode][['Description']][-1:]['Description'].iloc[0])        
            prod_desc_tup = productCode, ' - ', german_sale_trans_df_orig[german_sale_trans_df_orig['StockCode'] == productCode][['Description']][-1:]['Description'].iloc[0]
            prod_desc = ''.join(prod_desc_tup)
        
            top10product_content = get_recommendations_content(productCode)
            top10product_content_list = top10product_content['StockCode'].tolist()
            print('Top 10 StockCodes - Item Collab \n',top10product_collab_list)
            print('Top 10 StockCodes - Content based \n',top10product_content_list)
            prod_to_recommend = filter_top10(top10product_collab_list, top10product_content_list)
            if(len(prod_to_recommend) >= 1):
                dictList = printProducts(prod_to_recommend)
                res=dictList.to_dict('records')
                res.append(prod_desc)
                return res
            else:
                return fallbackPopularItems()
    except:
        return fallbackPopularItems()

app = Flask(__name__)
CORS(app)
@app.route('/api/popularItems',methods=['GET'])
def popularItems():
    dictList = printProducts(popularGermanItems())
    return jsonify(dictList.to_dict('records'))
  
@app.route('/api/contentBased',methods=['POST'])
def contentBased():
    try:
        data = request.get_json()
        stockCode = data['stockCode']
        print('\n Product Recomendations for ',stockCode, '-', german_sale_trans_df_orig[german_sale_trans_df_orig['StockCode'] == stockCode][['Description']][-1:]['Description'].iloc[0])
        prod_desc_tup = stockCode, ' - ', german_sale_trans_df_orig[german_sale_trans_df_orig['StockCode'] == stockCode][['Description']][-1:]['Description'].iloc[0]
        prod_desc = ''.join(prod_desc_tup)
        top10product_content = get_recommendations_content(stockCode)
        top10product_content_list = top10product_content['StockCode'].tolist()
        print('Top 10 StockCodes - Content based \n',top10product_content_list)
        res = top10product_content.to_dict('records')
        res.append(prod_desc)
        return jsonify(res)
    except:
        print('StockCode Not Recognised !!', '\n')
        print('Take a look at popular products.....', '\n')
        prod_desc = "StockCode Not Found - Below are the Popular items sold"
        dictList = printProducts(popularGermanItems())
        res = dictList.to_dict('records')
        res.append(prod_desc)
        return jsonify(res)
    
@app.route('/api/collabBasedItem', methods=['POST'])
def collabBasedItem():
    data = request.get_json()
    itemProductCode = data['stockCode']
    return jsonify(itemBasedCollabFilter(itemProductCode))
   
@app.route('/api/hybridBased', methods=['POST'])
def hyridBased():
    data = request.get_json()
    itemProductCode = data['stockCode']
    return jsonify(hybridBasedModel(itemProductCode))
    
@app.route('/api/collabBasedUser',methods=['POST'])
def collabBasedUser():
    data = request.get_json();
    userID = data['user_id']
    recommedItems = userBasedCollabFilter(userID)
    return jsonify(recommedItems)
    
#Server configured to run on port 8111
#"http://127.0.0.1:8111/api/"
if __name__ == '__main__':
    retail_dataframe = pd.read_excel("Online Retail.xlsx")
    germany_retail_df = retail_dataframe[retail_dataframe['Country']=='Germany']
    germany_retail_df = dataCleaning(germany_retail_df)
    german_sale_trans_df = getSaleData(germany_retail_df)
    german_sale_trans_df_orig = german_sale_trans_df.copy()
    german_sale_trans_df = dataPreprocessing(german_sale_trans_df)
    german_stock_desc_df = createStockDescMapperDF(german_sale_trans_df_orig)
    german_sale_trans_group_df = createCustomerStockQtyMapperDF(german_sale_trans_df)
    df_matrix_item = pd.pivot_table(german_sale_trans_group_df, values='Quantity', columns='CustomerID', index='StockCode').fillna(0)
    df_matrix_user = pd.pivot_table(german_sale_trans_group_df, values='Quantity', columns='StockCode', index='CustomerID').fillna(0)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
    temp_list=df_matrix_item.index.tolist()
    cust_item_features = csr_matrix(df_matrix_item.values)
    temp_user_list=df_matrix_user.index.tolist()
    cust_user_features = csr_matrix(df_matrix_user.values)
    app.run(port=8111, debug=True)