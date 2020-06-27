import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


def encoding(sentence):
    """
    Encoding sentence using BERT, SINGLE sentence version
    :param sentence: input sentence
    :return: 4D encoded layer, #layers * #batch(sentence) * #token * #hidden unit
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    marked_text = "[CLS] " + sentence + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model = BertModel.from_pretrained('bert-base-uncased')  # load pre-train model from google
    model.eval()  # evaluation mode

    # Predict hidden states features for each layer
    with torch.no_grad():  # no_grad = no gradient, save memory and speed up
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    return encoded_layers


def info(encode_layer):
    """
    Output status of encoded layer.
    :param encode_layer: BERT model
    """
    layer_i = 0
    batch_i = 0
    token_i = 0

    print("Number of layers:", len(encode_layer))
    print("Number of batches:", len(encode_layer[layer_i]))
    print("Number of tokens:", len(encode_layer[layer_i][batch_i]))
    print("Number of hidden units:", len(encode_layer[layer_i][batch_i][token_i]))


def check(encode_layer, layer_i=0, batch_i=0, token_i=0, n=10):
    """
    Checks token_i in our sentence, get its feature values form layer_i
    :param encode_layer: BERT model
    :param layer_i:      # of layer
    :param batch_i:      # of batch(sentence)
    :param token_i:      # of token
    :param n:            output first n values
    """
    vec = encode_layer[layer_i][batch_i][token_i]
    print(vec[:n])  # print first 10 values


if __name__ == "__main__":
    sentence1 = "Here is the sentence I want embeddings for."
    print("sentence: " + sentence1)
    embedding = encoding(sentence1)
    info(embedding)
    check(embedding)
    print("end")
