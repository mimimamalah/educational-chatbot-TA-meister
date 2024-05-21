import argparse
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from html import unescape
import json
import os


class Question:
    """
    Stores the question attributes, and a list of answers to this question.
    """
    def __init__(self, post_id:int, accepted_answer_id:int, answer_count:int, score:int, title:str, body:str):
        self.post_id = post_id
        self.accepted_answer_id = accepted_answer_id
        self.answer_count = answer_count
        self.score = score
        self.title = title
        self.body = body
        self.answers = []

    def __str__(self):
        return(
            f"#####################\n"
            f"Question {self.post_id}\n"
            f"#####################\n"
            f"accepted_answer_id: {self.accepted_answer_id}, answer_count: {self.answer_count}, found_answers: {len(self.answers)} score: {self.score}\n\n"
            f"TITLE:\n######\n{self.title}\n"
            f"BODY:\n#####\n{self.body}\n"
            + f"\n".join(str(x) for x in self.answers)
        )
    

class Answer:
    """
    Stores the answer attributes. And provide ordering based on the score.
    """
    def __init__(self, post_id:int, parent_id:int, score:int, body:int):
        self.post_id = post_id
        self.parent_id = parent_id
        self.score = score
        self.body = body

    def __lt__(self, other):
        return self.score < other.score
    
    def __eq__(self, other):
        return self.score == other.score

    def __str__(self):
        return (
            f"Answer {self.post_id}\n"
            f"#############\n"
            f"post_id: {self.post_id}, parent_id: {self.parent_id}, score: {self.score}\n\n"
            f"{self.body}"
        )
    

def read_xml(input_file: str) -> dict[int, Question]:
    """
    Read the un-processed data from an xml file and return a dictionary containing all the questions.
    """
    # The key is the post id and the value is the question associated with this id
    questions: dict[int, Question] = {}

    # Iterate on all the rows of the xml file
    tree = ET.parse(input_file)
    root = tree.getroot()
    for child in root:
        post = child.attrib

        # If the post is a question
        if post['PostTypeId'] == '1': 
            question = Question(post_id= int(post['Id']), 
                                accepted_answer_id= int(post.get('AcceptedAnswerId', '-1')), 
                                answer_count= int(post['AnswerCount']), 
                                score= int(post['Score']), 
                                title= post['Title'],
                                body = BeautifulSoup(unescape(post['Body']), 'html.parser').get_text()
                                )
            
            if question.post_id in questions: # We found an answer before the question
                previous_question = questions[question.post_id]
                assert(previous_question.answer_count == -1)
                print(f"Recovered question {question.post_id}")
                question.answers = previous_question.answers  # Recover previous answers

            questions[question.post_id] = question

        # If the post is an answer
        elif post['PostTypeId'] == '2':
            answer = Answer(post_id= int(post['Id']),
                            parent_id= int(post['ParentId']),
                            score= int(post['Score']),
                            body= BeautifulSoup(unescape(post['Body']), 'html.parser').get_text()
                            )
            
            if answer.parent_id not in questions:  # We found an anwer before the question
                # Add question to dictionary and mark it with special value answer_count = -1
                questions[answer.parent_id] = Question(post_id=answer.parent_id, accepted_answer_id=-1,
                                                       answer_count=-1, score=0, title='', body='')
                print(f"Warning: Found answer {answer.post_id} before its parent {answer.parent_id}")
            
            questions[answer.parent_id].answers.append(answer)

    return questions


def filter_questions(questions: dict[int, Question]) -> dict[int, Question]:
    """
    Filter out questions that do not have at least 2 answers or have all answers with the same score.
    """
    def is_valid_question(question: Question) -> bool:
        answers = question.answers
        if len(answers) < 2 or min(answers) == max(answers):
            return False
        
        return True

    return {q_id: question for q_id, question in questions.items() if is_valid_question(question)}


def validate_data(questions: dict[int, Question]):
    """
    Check if the data is valid. 
    This function will raise an exception if the data is not valid, or print a warning if it is not optimal.
    """
    for q_id, question in questions.items():
        assert(question.post_id == q_id)
        assert(question.answer_count >= 0)
        assert(question.title != '')
        assert(question.body != '')
        assert(len(question.answers) >= 2)
        assert(max(question.answers) > min(question.answers))
        for answer in question.answers:
            assert(answer.parent_id == q_id)
            assert(answer.body != '')
            assert(answer.post_id is not None)

        if question.answer_count != len(question.answers):
            print(
                f"Warning: answer_count does not match the number of answers found: "
                f"q_id: {q_id}, answer_count: {question.answer_count}, found_answers: {len(question.answers)}"
                )

        if question.accepted_answer_id not in [-1] + [a.post_id for a in question.answers]:
            print(
                f"Warning: accepted_answer_id is not a valid answer id: "
                f"q_id: {q_id}, accepted_answer_id: {question.accepted_answer_id}, "
                f"answer_count: {question.answer_count}, found_answers: {[a.post_id for a in question.answers]}")


def write_jsonl(questions: dict[int, Question], output_file: str):
    """
    Write a question dictionary to a json file, in the expected DPO format
    """
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'w') as f:
        for q_id, question in questions.items():
            chosen = max(question.answers)
            rejected = min(question.answers)

            f.write(
                json.dumps({
                    "prompt": question.title + ' ' + question.body,
                    "chosen": chosen.body,
                    "rejected": rejected.body
                }) + '\n'
            )
 

if __name__ == '__main__':
    # Parse program arguments
    parser = argparse.ArgumentParser(description='Data Cleanup Script')
    parser.add_argument('input', type=str, help='Path to input file')
    parser.add_argument('output', type=str, help='Path to output file')
    args = parser.parse_args()

    # Read unprocessed data, filter out invalid questions, and write to output file
    questions = read_xml(args.input)
    print("Read", len(questions), "questions\n")

    questions = filter_questions(questions)
    print("Filtered to", len(questions), "questions\n")

    validate_data(questions)
    print("Data validation finished\n")

    write_jsonl(questions, args.output)
    print("Data written to", args.output, "successfully\n")
