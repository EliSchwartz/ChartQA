import json
import os

# llava format
'''
{'id': '000000033471', 'image': '000000033471.jpg', 'conversations': [{'from': 'human', 
'value': '<image>\nWhat are the colors of the bus in the image?'}, 
{'from': 'gpt', 'value': 'The bus in the image is white and red.'}, 
{'from': 'human', 'value': 'What feature can be seen on the back of the bus?'}, 
{'from': 'gpt', 'value': 'The back of the bus features an advertisement.'}, 
{'from': 'human', 'value': 'Is the bus driving down the street or pulled off to the side?'}, 
{'from': 'gpt', 'value': 'The bus is driving down the street, which is crowded with people and other vehicles.'}]}
'''

# docvqa format
'''
{"imgname": "10095.png", "query": "Is the value of Favorable 38 in 2015?", "label": "Yes"}
'''

def convert_json(input_path, output_path):

    with open(input_path) as f:
        in_list = json.load(f)

    all_data = []
    for id, x in enumerate(in_list):
        converted_data = {
            'id': str(id),
            'image': x['imgname'],
            'conversations': [
                {'from': 'human', 'value': f'<image>\n{x["query"]}'},
                {'from': 'gpt', 'value': f'{x["label"]}'},  # GPT respones are actually the answers
            ]
        }
        all_data.append(converted_data)

    # Save the converted data as a JSON file
    with open(output_path, 'w') as output_file:
        json.dump(all_data, output_file, indent=2)
    print(f"Conversion complete. Saved as {output_path}")


if __name__ == "__main__":
    base_path = 'ChartQA Dataset'
    for split in ['train', 'val', 'test']:
        for generated_method in ['human', 'augmented']:
            input_path = os.path.join(base_path, split, f'{split}_{generated_method}.json')
            output_path = os.path.join(base_path, split, f'{split}_{generated_method}_llava.json')
            convert_json(input_path, output_path)
