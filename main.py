#importing decoder
from decoder.output_embeding import OutputEmbedding
from decoder.decoder_positionsl_encoding import PositionEncoding as Position_decoder
from decoder.mask_multi_head_attention import Mask_MultiHeadAttention
from decoder.decoder_layer_normalization import LayerNormalization_1
from decoder.cross_attention import CrossAttention
from decoder.layer_norm_at_cross import LayerNormalization_2
from decoder.decoder_feed_forward import TransformerFeedForwardLayer as feed_decoder
from decoder.layer_norm_at_feed import LayerNormalization_3

#importing encoder
from encoder.input_encoding import input_embedding
from encoder.postel_encoding import PositionEncoding as Position_encoder
from encoder.multi_head_attention import MultiHeadAttention
from encoder.layer_normalization import LayerNormalization_
from encoder.feed_forward import TransformerFeedForwardLayer as feed_encoder
from encoder.layer_norm_feed_ import LayerNormalization


class Main:
    def __init__(self, num = 6, input = None, output = None):
        self.num_of_decoder = num
        self.input = input
        self.output = output
    
    def create(self):
        self.genetated_output = None
        x_decoder = OutputEmbedding(self.output).get_encoded_vector()
        x_decoder_pos = Position_decoder(x_decoder).get_output()
        x_encoder = input_embedding(self.input).get_encoded_vector()
        x_encoder_pos = Position_encoder(x_encoder).get_output()
        for _ in range(self.num_of_decoder):

            #encoder
            x__encoder_attention = MultiHeadAttention(x_encoder_pos).get_output()
            x_encoder_norm = LayerNormalization_(attention_vector=x__encoder_attention, pos_vector=x_encoder_pos).normalize()
            x_encoder_feed = feed_encoder().call(x_encoder_norm)
            x__encoder_fnorm = LayerNormalization(feed_vector=x_encoder_feed, norm_vector=x_encoder_norm).normalize()
            x_encoder_pos = x__encoder_fnorm

            #decoder
            x_decoder_mask = Mask_MultiHeadAttention(x_decoder_pos).get_output()
            x_decoder_norm = LayerNormalization_1(position_vector=x_decoder_pos, mask_vector=x_decoder_mask).normalize()
            x_decoder_cross = CrossAttention(input_vectors=x__encoder_fnorm, output_vectors=x_decoder_norm).get_output()
            x_decoder_cnorm = LayerNormalization_2(norm_vector=x_decoder_norm, cross_vector=x_decoder_cross).normalize()
            x_decoder_feed = feed_decoder().call(x_decoder_cnorm)
            x_decoder_fnorm = LayerNormalization_3(feed_vector=x_decoder_feed, norm_vector=x_decoder_cnorm)
            x_decoder_pos = x_decoder_fnorm
        self.genetated_output = x_decoder_fnorm
    
    def get_output(self):
        return self.output


