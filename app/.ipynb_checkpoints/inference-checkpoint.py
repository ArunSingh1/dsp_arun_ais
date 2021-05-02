import pandas as pd

def submission_to_csv(classifier, test_df):
    
    y_pred = classifier.predict(test_df)
    ID=test_df['Id']
    df_submission = pd.DataFrame({'Id': ID, 'SalePrice': y_pred})
    
    df_submission.to_csv('submission.csv',index=False)
    return df_submission.head()
