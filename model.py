class LandmarkRetrievelModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(LandmarkRetrievelModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.n_features = self.model.classifier.in_features
        self.model.reset_classifier(0)
        self.fc = nn.Linear(self.n_features, CONFIG['num_classes'])

    def forward(self, x):
        features = self.model(x)    # features = (bs, embedding_size)
        output = self.fc(features)  # outputs = (bs, num_classes)
        return output
    
    def extract_features(self, x):
        features = self.model(x)    # features = (bs, embedding_size)
        return features
    
model = LandmarkRetrievelModel(CONFIG['model_name'])
model.to(CONFIG['device']);
