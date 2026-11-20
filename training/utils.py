def freeze_bert_params(model, freeze_first_encoder_num: int=6):
    """Freeze the embeddings and the first `freeze_first_encoder_num` encoder layers."""
    for param in model.model.embeddings.parameters():
        param.requires_grad = False

    for layer_num, layer in enumerate(model.model.layers):
        if layer_num < freeze_first_encoder_num:
            for param in layer.parameters():
                param.requires_grad = False
    return model
