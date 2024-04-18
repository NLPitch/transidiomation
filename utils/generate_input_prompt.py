
def create_prompt_to_model(n:int, source_text:str, source_language:str, target_language:str) -> list:
    match n:
        case 0:
            return create_zero_shot_prompt
        case 1:
            return create_one_shot_prompt 
        case 2:
            return create_two_shot_prompt
        case _:
            return create_five_shot_prompt
    
    return 0

def create_zero_shot_prompt(source_text:str, source_language:str, target_language:str) -> list:
    prompt = f"Translate {source_text} from {source_language} to {target_language}"

    message = [{"role": "user", "content": prompt}]
    return message

def create_one_shot_prompt(source_text:str, source_language:str, target_language:str) -> list:

    return 0

def create_two_shot_prompt(source_text:str, source_language:str, target_language:str) -> list:

    return 0

def create_five_shot_prompt(source_text:str, source_language:str, target_language:str) -> list:

    return 0