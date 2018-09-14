# Some separate code for turning word2vec gensim model into an embedding dict used in demo visalisation of word embeddings
import sys
import gensim
import pickle
import datetime
import numpy as np

def convert_gensim_word2vec_model_to_embedding_file(path):
    model = gensim.models.Word2Vec.load(path)

    vocab = list(model.wv.vocab.keys())
    # self.embeddings_dict = {word:self.model1[word] for word in self.vocab}
    embeddings_array = np.concatenate([model[word].reshape(1, -1) for word in vocab], axis=0)

    embeddings_file_to_save = {'labels': vocab, 'embeddings': embeddings_array}
    save_fp = 'model_files/embeddings_obj_{}_{}.pkl'.format(len(vocab), datetime.datetime.now().strftime(
                                                                                          "%Y-%m-%d_%H:%M:%S"))
    with open(save_fp, 'wb') as handle:
        pickle.dump(embeddings_file_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved file to path: ', save_fp)

if len(sys.argv) < 2:
    sys.stderr.write('Usage: {} <Word2Vec model file>\n'.format(sys.argv[0]))
    sys.exit()
convert_gensim_word2vec_model_to_embedding_file(sys.argv[1])
