from project_utils import *
import config as cfg

import time
import glob
import torch
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import json

layer = ['classifier', '4']

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def make_sets(img_classes, val_ratio):
  #img_classes = np.array([dataset[x][1] for x in range(len(dataset))])

  classes_unique, classes_count = np.unique(img_classes, return_counts=True)
  print(classes_unique)
  print(classes_count)

  if sum(classes_count == classes_count[0]) == len(classes_unique):
    n_imgs = classes_count[0]
  else:
    print("Error: unequal number of imgs per class")

  val_size = int(n_imgs * val_ratio)
  train_size = n_imgs - val_size
  n_rounds = int(1 / val_ratio)

  ixs = np.arange(len(img_classes))

  samps_train = np.full((n_rounds, len(classes_unique), train_size), np.nan)
  samps_val = np.full((n_rounds, len(classes_unique), val_size), np.nan)
  for c in range(len(classes_unique)):
    c_ix = ixs[img_classes == classes_unique[c]]

    np.random.seed(c)
    np.random.shuffle(c_ix)

    for r in range(n_rounds):
      start = r * val_size
      end = (r + 1) * val_size

      samps_val[r,c] = c_ix[start:end]
      samps_train[r,c] = np.array([x for x in c_ix if not x in samps_val[r,c]])

  return(samps_train, samps_val)


if __name__ == '__main__':
    layer = cfg.cnn.layer
    # get_features(layer)

    start = time.time()

    data_dir = cfg.cnn.stim_dir
    out_dir = cfg.cnn.out_dir
#    data_dir = '/content/drive/MyDrive/phd/project1/kar_version/stimuli/all'
#    out_dir = '/content/drive/MyDrive/phd/project1/kar_version/output_files/'
    img_files = glob.glob(data_dir + "/**/*.png", recursive=True)

    preprocess = transforms.Compose([
    #        transforms.Resize(256),
    #        transforms.CenterCrop(224),
        transforms.ToTensor(),
    #        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])

    net = models.alexnet(pretrained=True)
    net.eval()   # Set model to evaluate mode
    print(net.features._modules)

    if layer[0] == 'features':
        net.features._modules[layer[1]].register_forward_hook(get_activation(layer[1]))
        out_features = net.features._modules[layer[1]].out_features
    elif layer[0] == 'classifier':
        net.classifier._modules[layer[1]].register_forward_hook(get_activation(layer[1]))
        out_features = net.classifier._modules[layer[1]].out_features

    features_set = np.full((len(img_files),out_features), np.nan)
    img_ids = []
    for i in range(len(img_files)):
        _, cat, img_ix = [x for x in str.split(img_files[i][len(data_dir):-4],'/')]
        img_id = cat + '_' + img_ix

        print('getting activations: ' + img_id)

        img = Image.open(img_files[i]).convert('RGB')
        imgt = preprocess(img)
        ibatch = imgt.unsqueeze(0)

        activation = {}
        with torch.no_grad():
            out = net(ibatch)

        features = activation[layer[1]].numpy().flatten()
        features_set[i] = features
        img_ids.append(img_id)
        #activation_set[layer_label].update({img_id: features})

    activation_set = {
                      'id': np.array(img_ids),
                      'features': features_set
                      }

    out_file = cfg.cnn.features_file
    print('writing file: ' + out_file)
    with open(out_file, 'wb') as f:
      pickle.dump(activation_set, f, pickle.HIGHEST_PROTOCOL)



    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = activation_set['features'].copy()

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=1000)
    features_pca = pca.fit_transform(x


    pca_set = {
          'id': activation_set['id'],
          'components': features_pca
           }

    pca_file = cfg.cnn.pca_file
    print('writing file: ' + pca_file)
    with open(pca_file, 'wb') as f:
        pickle.dump(pca_set, f, pickle.HIGHEST_PROTOCOL)


    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score

    val_ratio = .25
    dshape = pca_set['id'].shape

    obj_id, img_id = zip(*[tuple(x.split('_')) for x in pca_set['id']])
    objs = np.unique(obj_id)

    trainsets_ix, testsets_ix = make_sets(np.array(obj_id), val_ratio)

    n_rounds = int(1 / val_ratio)

    DA = np.full((dshape[0], len(objs)), np.nan)
    ids = []
    imgs_list = []
    objs_list = []
    for round in range(n_rounds):
      print('round ', round + 1, 'out of ', n_rounds)
      train_ix = trainsets_ix[round].flatten().astype(int)
      test_ix = testsets_ix[round].flatten().astype(int)

      train_x = pca_set['components'][train_ix]
      train_y = np.array(obj_id)[train_ix]

      test_x = pca_set['components'][test_ix]
      test_y = np.array(obj_id)[test_ix]

      classifier = LinearSVC(penalty = 'l2',
                              loss = 'hinge',
                              C = .5,
                              multi_class = 'ovr',
                              fit_intercept = True,
                              max_iter = 100000)
      classifier.fit(train_x, train_y)

      R = np.transpose(test_x)
      bias = [[x] for x in classifier.intercept_]

      sc = np.dot(classifier.coef_, R) + bias
      sc = np.transpose(sc)
      p_obj = np.array([np.exp(x)/sum(np.exp(x)) for x in sc]) #softmax

      start = round * len(test_ix)
      end = (round + 1) * len(test_ix)

      DA[start:end] = p_obj
      ids.append(pca_set['id'][test_ix])
      imgs_list.append(np.array(img_id)[test_ix])
      objs_list.append(test_y)

    da_dict = {
        'da': DA,
        'id': np.array(ids),
        'obj': np.array(objs_list),
        'img': np.array(imgs_list)
    }

    da_file = cfg.cnn.da_file
    print('writing file: ' + da_file)
    with open(da_file, 'wb') as f:
      pickle.dump(da_dict, f, pickle.HIGHEST_PROTOCOL)
