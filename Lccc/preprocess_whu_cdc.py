import argparse
import json
import os


SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<UNK>': 1,
    '<START>': 2,
    '<END>': 3,
}


def tokenize(s, delim=' ', add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = [token for token in s.split(delim) if token]
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, min_token_count=1):
    token_to_count = {}
    for _, token_lists in sequences:
        for seq in token_lists:
            for token in seq:
                token_to_count[token] = token_to_count.get(token, 0) + 1

    token_to_idx = dict(SPECIAL_TOKENS)
    for token, count in sorted(token_to_count.items()):
        if token in token_to_idx:
            continue
        if count > min_token_count:
            token_to_idx[token] = len(token_to_idx)
    return token_to_idx


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    token_dir = os.path.join(args.save_dir, 'tokens')
    os.makedirs(token_dir, exist_ok=True)

    for split in ('train', 'val', 'test'):
        list_file = os.path.join(args.save_dir, split + '.txt')
        if os.path.exists(list_file):
            os.remove(list_file)

    with open(args.caption_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    max_length = -1
    all_cap_tokens = []
    for img in data['images']:
        tokens_list = []
        for caption in img['sentences']:
            raw = caption['raw']
            assert len(raw) > 0, 'error: some image has no caption'
            tokens = tokenize(
                raw,
                add_start_token=True,
                add_end_token=True,
                punct_to_keep=[';', ','],
                punct_to_remove=['?', '.'],
            )
            tokens_list.append(tokens)
            max_length = max(max_length, len(tokens))
        all_cap_tokens.append((img['filename'], tokens_list))

        stem = os.path.splitext(img['filename'])[0]
        with open(os.path.join(token_dir, stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokens_list))

        split = img.get('split') or stem.split('_')[0]
        if split in {'train', 'val', 'test'}:
            with open(os.path.join(args.save_dir, split + '.txt'), 'a', encoding='utf-8') as f:
                f.write(img['filename'] + '\n')

    vocab = build_vocab(all_cap_tokens, args.word_count_threshold)
    with open(os.path.join(args.save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f)

    print('WHU_CDC preprocessing done.')
    print('max_length of the dataset:', max_length)
    print('vocab size:', len(vocab))
    print('save_dir:', args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_json', default='./whu_CDC_dataset/whuCCcaptions.json')
    parser.add_argument('--save_dir', default='./data/WHU_CDC/')
    parser.add_argument('--word_count_threshold', default=5, type=int)
    main(parser.parse_args())
