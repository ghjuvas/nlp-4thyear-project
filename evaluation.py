'''
Evaluation script for NLP project (ABSA on Restaurant category)
'''
import argparse
import os


# checking paths of the files
def check_path_exists(path: str) -> None:
    '''
    Check if path to file exists.
    '''
    if not os.path.exists(path):
        raise ValueError('Path is not valid!')


def check_file_extension(path: str) -> None:
    '''
    Check extension of file.
    Raise error if file extension is not in the list
    of the allowed extensions.
    '''
    if os.path.splitext(path)[1] != '.txt':
        raise ValueError(
            f'''Program do not support this file extension!
            Please, use .txt'''
            )


def get_gold_info(gold_path: str) -> tuple:
    '''
    Get gold aspect categories and size of the gold standard from file.
    '''
    gold_aspect_cats = {}

    with open(gold_path) as fg:
        for line in fg:
            line = line.rstrip('\r\n').split('\t')
            if line[0] not in gold_aspect_cats:
                gold_aspect_cats[line[0]] = {"starts":[], "ends":[], "cats":[], "sents":[]}
            gold_aspect_cats[line[0]]["starts"].append(int(line[3]))
            gold_aspect_cats[line[0]]["ends"].append(int(line[4]))
            gold_aspect_cats[line[0]]["cats"].append(line[1])
            gold_aspect_cats[line[0]]["sents"].append(line[5])

    gold_size = sum([len(gold_aspect_cats[x]["cats"]) for x in gold_aspect_cats])

    return gold_aspect_cats, gold_size


def compute_match(pred_path: str, gold_aspect_cats: dict) -> tuple:
    '''
    Compute match between predicted and gold aspects.
    '''

    full_match, partial_match, full_cat_match, partial_cat_match = 0, 0, 0, 0
    total = 0
    fully_matched_pairs = []
    partially_matched_pairs = []

    with open(pred_path) as fp:
        for line in fp:
            total += 1
            line = line.rstrip('\r\n').split('\t')
            start, end = int(line[3]), int(line[4])
            category = line[1]
            doc_gold_aspect_cats = gold_aspect_cats[line[0]]
            if start in doc_gold_aspect_cats["starts"]:
                i = doc_gold_aspect_cats["starts"].index(start)
                if doc_gold_aspect_cats["ends"][i] == end:
                    full_match += 1
                    if doc_gold_aspect_cats["cats"][i] == category:
                        full_cat_match += 1
                    else:
                        partial_cat_match += 1
                    fully_matched_pairs.append(
                        (
                            [
                                doc_gold_aspect_cats["starts"][i], 
                                doc_gold_aspect_cats["ends"][i], 
                                doc_gold_aspect_cats["cats"][i],
                                doc_gold_aspect_cats["sents"][i]
                            ],
                            line
                        )
                    )
                    continue
            for s_pos in doc_gold_aspect_cats["starts"]:
                if start <= s_pos:
                    i = doc_gold_aspect_cats["starts"].index(s_pos)
                    if doc_gold_aspect_cats["ends"][i] == end:
                        partial_match += 1
                        partially_matched_pairs.append(
                            (
                                [
                                    doc_gold_aspect_cats["starts"][i], 
                                    doc_gold_aspect_cats["ends"][i], 
                                    doc_gold_aspect_cats["cats"][i],
                                    doc_gold_aspect_cats["sents"][i]
                                ],
                                line
                            )
                        )
                        if doc_gold_aspect_cats["cats"][i] == category:
                            partial_cat_match += 1
                        continue
                    matched = False
                    for e_pos in doc_gold_aspect_cats["ends"][i:]:
                        if s_pos <= end <= e_pos:
                            partial_match += 1
                            partially_matched_pairs.append(
                                (
                                    [
                                        doc_gold_aspect_cats["starts"][i], 
                                        doc_gold_aspect_cats["ends"][i], 
                                        doc_gold_aspect_cats["cats"][i],
                                        doc_gold_aspect_cats["sents"][i]
                                    ],
                                    line
                                )
                            )
                            if doc_gold_aspect_cats["cats"][i] == category:
                                partial_cat_match += 1
                            matched = True
                            break
                    if matched:
                        break
                if start > s_pos:
                    i = doc_gold_aspect_cats["starts"].index(s_pos)
                    if start < doc_gold_aspect_cats["ends"][i] <= end:
                        partial_match += 1
                        partially_matched_pairs.append(
                            (
                                [
                                    doc_gold_aspect_cats["starts"][i], 
                                    doc_gold_aspect_cats["ends"][i], 
                                    doc_gold_aspect_cats["cats"][i],
                                    doc_gold_aspect_cats["sents"][i]
                                ],
                                line
                            )
                        )
                        if doc_gold_aspect_cats["cats"][i] == category:
                            partial_cat_match += 1
                        break

    return full_match, partial_match, full_cat_match, partial_cat_match, fully_matched_pairs, partially_matched_pairs, total


def compute_sentiment_accuracy(matches: list) -> float:
    '''
    Compute sentiment accuracy on the matches.
    '''
    matched_sentiment = 0.

    for pair in matches:
        *_, gold_s = pair[0]
        *_, pred_s = pair[1]
        if gold_s == pred_s:
            matched_sentiment += 1

    return matched_sentiment


def compute_overall_sentiment_accuracy(gold_cats_path: str, pred_cats_path: str) :
    '''
    Compute overall sentiment accuracy.
    '''
    with open(gold_cats_path) as gc, open(pred_cats_path) as pc:
        gold_labels = set(gc.readlines())
        pred_labels = set(pc.readlines())

    return len(gold_labels & pred_labels) / len(gold_labels)


def main() -> None:
    '''
    Parse arguments and run evaluation.
    '''
    parser = argparse.ArgumentParser()

    evaluation_mode = parser.add_mutually_exclusive_group(required=True)
    evaluation_mode.add_argument(
        '-acatsent', nargs=2,
        help='compute quality measures on the aspect terms category and sentiment'
    )
    evaluation_mode.add_argument(
        '-rcatsent', nargs=2,
        help='compute quality measures on the review category'
    )

    args = parser.parse_args()

    if args.acatsent:
        gold_path, pred_path = args.acatsent

        check_path_exists(gold_path)
        check_file_extension(gold_path)
        check_path_exists(pred_path)
        check_file_extension(pred_path)

        gold_aspect_cats, gold_size = get_gold_info(gold_path)
        full_match, part_match, full_cat_match, part_cat_match, full_matches, part_matches, total = compute_match(pred_path, gold_aspect_cats)
        full_sent_match = compute_sentiment_accuracy(full_matches)
        part_sent_match = compute_sentiment_accuracy(part_matches)

        print(f"""
            Full match precision: {full_match / total}
            Full match recall: {full_match / gold_size}

            Partial match ratio in pred: {(full_match + part_match)  / total}

            Full category accuracy: {full_cat_match / total}
            Partial category accuracy: {(full_cat_match + part_cat_match) / total}

            Full sentiment accuracy: {full_sent_match / len(full_matches)}
            Partial sentiment accuracy: {part_sent_match / len(part_matches)}
            """)

    if args.rcatsent:
        gold_path, pred_path = args.rcatsent

        check_path_exists(gold_path)
        check_file_extension(gold_path)
        check_path_exists(pred_path)
        check_file_extension(pred_path)

        overall_sentiment_accuracy = compute_overall_sentiment_accuracy(gold_path, pred_path)
        print(f'Overall sentiment accuracy: {overall_sentiment_accuracy}')


if __name__ == '__main__':
    main()
