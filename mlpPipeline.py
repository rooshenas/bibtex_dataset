import os
import pandas as pd

from sklearn.metrics import hamming_loss

from d3m.container.dataset import D3MDatasetLoader, Dataset, CSVLoader

from common_primitives.denormalize import DenormalizePrimitive, Hyperparams as hyper_Den
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive, Hyperparams as hyper_Dat
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive, Hyperparams as hyper_Ext

from dsbox.spen.application.MLPClassifier import MLCHyperparams, Params, MLClassifier
from dsbox.datapreprocessing.cleaner.to_numeric import ToNumeric, Hyperparams as hyper_Nu
from dsbox.datapreprocessing.cleaner.encoder import Encoder, EncHyperparameter as hyper_En

h0 = hyper_Den.defaults()
h1 = hyper_Dat.defaults()
primitive_0 = DenormalizePrimitive(hyperparams = h0)
primitive_1 = DatasetToDataFramePrimitive(hyperparams = h1)

dataset_train_file_path = 'bibtex_dataset/bibtex_dataset/datasetDoc.json'
dataset = D3MDatasetLoader()

dataset_train = dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=os.path.abspath(dataset_train_file_path)))
dataset_org = primitive_0.produce(inputs = dataset_train)
res_df = primitive_1.produce(inputs = dataset_org.value)

h2 = hyper_Ext(                            {
                                    'semantic_types': (
                                        'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                        'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                    'use_columns': (),
                                    'exclude_columns': ()
                                })
h3 = hyper_Ext(                            {
                                    'semantic_types': (
                                        'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                        'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                    'use_columns': (),
                                    'exclude_columns': ()
                                })
primitive_2 = ExtractColumnsBySemanticTypesPrimitive(hyperparams=h2)
primitive_3 = ExtractColumnsBySemanticTypesPrimitive(hyperparams=h3)

attributes = primitive_2.produce(inputs=res_df.value)
target = primitive_3.produce(inputs=res_df.value)

primitive_5 = Encoder(hyperparams=hyper_En.defaults())

primitive_5.set_training_data(inputs=attributes.value)
primitive_5.fit()
encoded = primitive_5.produce(inputs=attributes.value)

primitive_7 = ToNumeric(hyperparams=hyper_Nu.defaults())

tonu = primitive_7.produce(inputs = encoded.value)

h = MLCHyperparams.defaults()
primitive = MLClassifier(hyperparams=h)

primitive.set_training_data(inputs = tonu.value, outputs = target.value)
primitive.fit()
res = primitive.produce(inputs=tonu.value)


y_pred = res.value.drop(columns=["d3mIndex"])
y_truth = target.value.drop(columns=["d3mIndex"])
hammingLoss = hamming_loss(y_truth, y_pred)
print('Hamming Loss on test data:', hammingLoss)

# saving the predictions.csv file
res.value.to_csv('predictions.csv')

# saving the scores.csv file
df = pd.DataFrame(columns=['metric', 'value'])
df.loc[len(df)] = ['hammingLoss', hammingLoss]
df.to_csv('scores.csv')
