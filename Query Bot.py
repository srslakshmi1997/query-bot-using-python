def lookup_query_from_database(self, params, context, node_id):
    queries_df = pd.read_csv('GenieDB.csv',index_col='Query',delimiter = ',',encoding = "ISO-8859-1")
    raw_list = queries_df.index.values.tolist()
    raw_list_lower = list(map(lambda x : x.lower().strip(), raw_list))
    sent_tokens = raw_list_lower
    lemmer = nltk.stem.WordNetLemmatizer()

    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]
        
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        
    def LemNormalize(text):
        return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
        
    def response(user_response):
        sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx_list = np.array(vals.argsort()[0])
        idx = idx_list[vals[0][idx_list]>0.40][-4:-1]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if req_tfidf==0 or len(idx)==0:
            robo_response="I couldnt find anything in my Database"
            return robo_response
        else:
            robo_response = []
            for ind in reversed(idx) :
                print("__________",sent_tokens[ind],vals[0][ind],"___________")
                robo_response.append(sent_tokens[ind])
            return robo_response
            
        
    user_response = input("Enter the query : ")
    user_response=user_response.lower()
    inter_res_list = response(user_response)
        
    if inter_res_list != "I couldnt find anything in my Database":
        print_result = {}
        orderv = 1
        order = 'pri1'
        for inter_res in inter_res_list:
            var_result = {}
            index_result = raw_list_lower.index(inter_res)
            var_result_dict = {'Solution': queries_df.loc[raw_list[index_result]]['Solution'], 'Links' : queries_df.loc[raw_list[index_result]]['Links'], 'Tags' : queries_df.loc[raw_list[index_result]]['Tags']}
            var_result['Question'] = raw_list[index_result]
            for var_val in var_result_dict:
                if str(var_result_dict[var_val]) not in 'nan':
                    var_result[var_val] = var_result_dict[var_val]
            print_result[order] = var_result
            orderv += 1
            order = 'pri'+str(orderv) ##### Sub Dict Spike #######
        return print_result
    else : 
        sent_tokens.remove(user_response)
        return inter_res_list
