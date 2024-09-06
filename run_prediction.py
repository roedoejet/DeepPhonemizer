from dp.phonemizer import Phonemizer
import unicodedata
import re
if __name__ == '__main__':

    checkpoint_path = 'checkpoints_utts_phones/model_step_10k.pt'
    phonemizer = Phonemizer.from_checkpoint(checkpoint_path)
    text = unicodedata.normalize('NFC', 'EWENE SÁN E TŦE U, MEQ EȽTÁLṈEW̱ Ȼ SNI,S SQÍEŦ E TŦE XĆṈINS. U, XENENEȻEL TŦE U, MEQ EȽTÁLṈEW̱ E Ȼ SI,ÁM,TEṈS. ĆŚḰÁLEȻEN TŦE U, MEQ SÁN. Í, Ȼ S,Á,ITEṈS TŦE U, MEQ SÁN X̱EN,IṈ E TŦE SĆÁ,ĆE,S.')
    # text = unicodedata.normalize('NFC', 'W̱SÁNEĆ')
    # text = list(unicodedata.normalize('NFC', 'ȻELEȻENSISEṈ SW̱'))
    # text = unicodedata.normalize('NFC', 'SENĆOŦEN')
    # text = [unicodedata.normalize('NFC', x) for x in "SḰÁL E TŦE MEQ SȻÁĆEL".split()]
    
    result = phonemizer.phonemise_list([text], lang='str')
    print(result.phonemes)
    for text, pred in result.predictions.items():
        tokens, probs = pred.phoneme_tokens, pred.token_probs
        for o, p in zip(tokens, probs):
            print(f'{o} {p}')
        tokens = ''.join(tokens)
        print(f'{text} | {tokens} | {pred.confidence}')

