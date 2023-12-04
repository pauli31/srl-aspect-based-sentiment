import json
import argparse
from tqdm import tqdm
import spacy

business_file = './../yelp_academic_dataset_business.json'
reviews_file = './../yelp_academic_dataset_review.json'
save_path = './../export/'

def load_bussines():

    fn = business_file
    bussines = {}
    categories = set()
    with open(fn) as data_file:
        counter = 0
        for line in data_file:
            counter += 1
            json_res = json.loads(line)
            id = json_res['business_id']
            bussines[id] = json_res
            categories_res = json_res['categories']
            if categories_res != None:
                categories_res = categories_res.split(',')
                categories_res = [word.strip() for word in categories_res]
                categories.update(categories_res)

    return bussines, categories
    print("")

 # based on https://github.com/deepopinion/domain-adapted-atsc/blob/master/prepare_restaurant_reviews.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate finetuning corpus for restaurants.')

    parser.add_argument('--large',
                        action='store_true',
                        help='export large corpus (10 mio), default is 1 mio')

    parser.add_argument('--unlimited',
                        action='store_true',
                        help='if set, the --large parameter is overwritten, and all reviews are exported')

    args = parser.parse_args()

    # max_sentences = int(500)
    # review_limit = int(500)
    max_sentences = int(10e5)
    review_limit = int(150000)
    if args.large:
        review_limit = int(1500000)  # for 10 Mio Corpus
        max_sentences = int(10e6)  # for 10 Mio corpus

    if args.unlimited:
        review_limit = -1  # for 10 Mio Corpus
        max_sentences = -1  # for 10 Mio corpus


    bussines, categories = load_bussines()
    # python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('sentencizer')
    fn = reviews_file
    reviews = []
    texts = []
    # 4 million reviews to generate about minimum 10 mio sentences
    with open(fn) as data_file:
        counter = 0
        for line in data_file:
            counter += 1
            json_res = json.loads(line)
            id_review = (json_res['review_id'])
            id_business = json_res['business_id']
            text = json_res['text']
            texts.append(text)
            reviews.append((id_review, id_business, text))
            if counter == review_limit and review_limit > 0:
                break


    # get sentence segemented review with #sentences > 2
    def sentence_segment_filter_docs(doc_array):
        sentences = []

        for doc in nlp.pipe(doc_array, disable=['parser', 'tagger', 'ner'], batch_size=1000, n_process=6):
            sentences.append([sent.text.strip() for sent in doc.sents])

        return sentences


    print(f'Found {len(texts)} restaurant reviews')

    # print(f'Tokenizing Restaurant Reviews...')

    # sentences = sentence_segment_filter_docs(texts)
    # nr_sents = sum([len(s) for s in sentences])
    # print(f'Segmented {nr_sents} restaurant sentences')

    # tohle zakomentovat kdyz sentencese
    sentences = [review[2] for review in reviews]
    not_two_sentences = True

    # Save to file
    fn_out = save_path + f'yelp-reviews-max-sent_{max_sentences}.txt'
    sent_shorter = 0
    with open(fn_out, "w") as f:
        sent_count = 0
        for sents in tqdm(sentences):
            real_sents = []
            if not_two_sentences is True:
                iteration = [sents]
            for s in iteration:
                x = s.replace(' ', '').replace('\n', '').replace('\u200d', '').replace('\u200b', '')
                if x != '':
                    if s == "By far the best Avacado bread I have ever had.":
                        print(sents)
                        pass
                    real_sents.append(s.replace('\n', '').replace('\u200d', '').replace('\u200b', ''))
            if len(real_sents) >= 2 or not_two_sentences:
                sent_count += len(real_sents)
                if not_two_sentences is True:
                    str_to_write = " ".join(real_sents) + "\n"
                else:
                    str_to_write = "\n" + "\n".join(real_sents) + "\n"
                f.write(str_to_write)
            else:
                sent_shorter += 1

            if sent_count >= max_sentences and max_sentences > 0:
                break

    print(f'Done writing to {fn_out}')
    print(f'Sentences short than two: {sent_shorter}')
