import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import h5py
import io
import matplotlib.pyplot as plt

class SkinCancerDataset(Dataset):
    def __init__(self, hdf5_image_path, metadata_path, transform = None):
        self.hdf5_image_path = hdf5_image_path
        self.metadata = self.load_metadata(metadata_path)
        self.metadata = self.metadata_processing(self.metadata)
        self.images = self.load_image(hdf5_image_path)
        self.transform = transform
        
    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path,low_memory=False)
    
    def metadata_processing(self, metadata):
        #Drop redundant columns
        redundant_column = ["lesion_id","iddx_full","iddx_1","iddx_2","iddx_3","iddx_4","iddx_5","mel_mitotic_index","mel_thick_mm","tbp_lv_dnn_lesion_confidence","tbp_lv_location_simple","attribution","copyright_license"]
        index_column = ['isic_id', 'patient_id']
        metadata = metadata.drop(columns=redundant_column)
        metadata = metadata.drop(columns = index_column)
        
        #Label Encoding
        enc = LabelEncoder()
        ##1. Sex
        metadata.fillna({'sex': 'unknown'}, inplace=True)
        metadata['sex_enc'] = enc.fit_transform(metadata.sex.astype('str'))
        
        ##2. Age approx
        metadata['age_approx'] = metadata['age_approx'].fillna(metadata['age_approx'].mode().values[0])
        
        ##3. Site general
        metadata.fillna({'anatom_site_general': 'unknown'}, inplace=True)
        metadata['anatom_site_enc'] = enc.fit_transform(metadata.anatom_site_general.astype('str'))
        
        #OneHot Encoding
        onehot = OneHotEncoder(sparse_output = False)
        categorical_columns = metadata.select_dtypes(include=['object', 'category']).columns
        encoded_array = onehot.fit_transform(metadata[categorical_columns])
        encoded_df = pd.DataFrame(encoded_array, columns=onehot.get_feature_names_out(categorical_columns))
        metadata = pd.concat([metadata.drop(columns=categorical_columns), encoded_df], axis=1)

        return metadata

        
    def load_image(self, hdf5_image_path):
        image_list = []
        with h5py.File(hdf5_image_path, 'r') as hdf:
            image_keys = list(hdf.keys())
            for i,key in enumerate(image_keys[:]):
                image_data = hdf[key][()]
                image_list.append(image_data)
        return image_list
    
    def __len__(self):
        return self.metadata.shape[0]
    
    def __getitem__(self, index):

        # Convert the byte string to an image
        image = Image.open(io.BytesIO(self.images[index]))
        
        if self.transform:
            image = self.transform(image)
            
        metadata = self.metadata.iloc[index,2:-1].values
        target = self.metadata.iloc[index,1]
        
        metadata = torch.tensor(metadata, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        return image, metadata, target