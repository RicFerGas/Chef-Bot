# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import random
from nltk.stem import WordNetLemmatizer
# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load the data
data = pd.read_csv('df_filtred_clean_vegetarian.csv')

#Load the embeddings
import pickle
#Load embeddings from disc
with open("embeddings_cleanData.pkl", "rb") as fIn:
    stored_data = pickle.load(fIn)
    sentences = stored_data['sentences']
    embeddings = stored_data['embeddings']
# Function to lemmatize an ingredient
lemmatizer = WordNetLemmatizer()
def lemmatize_ingredient(ingredient):
    return lemmatizer.lemmatize(ingredient.lower())


    
class ActionConfirmIngredients(Action):
    def name(self) -> Text:
        return "action_confirm_ingredients"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
         # Extract new ingredients from the user's message
        new_ingredients = tracker.get_slot("ingredients")       # Retrieve the current value of the ingredients slot
        if new_ingredients is None:
            dispatcher.utter_message(text="I didn't get the ingredients. Can you please provide them again?")
            return []
        current_ingredients = tracker.get_slot("ingredients_list") or []


        new_ingredients = current_ingredients + new_ingredients 

        # Check the ingredients to ensure there are no duplicates or sub-words
        filtered_ingredients = []
        for ingredient in current_ingredients + new_ingredients:
            if ingredient not in filtered_ingredients:
                # Check if the ingredient is a sub-word of another ingredient
                is_sub_word = False
                for other_ingredient in current_ingredients + new_ingredients:
                    if ingredient != other_ingredient and ingredient in other_ingredient:
                        is_sub_word = True
                        break
                if not is_sub_word:
                    filtered_ingredients.append(ingredient)        
       # Send a message with the list of ingredients
        dispatcher.utter_message(text=f"So you want me to look for a recipie with this ingredients: {', '.join(filtered_ingredients)}, is that correct?")
        # Set the updated list of ingredients to the slot
        return [SlotSet("ingredients_list", filtered_ingredients)]
class ActionisVegetarian(Action):
    def name(self) -> Text:
        return "action_is_vegetarian"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        #Check last intent to see if the user accept or deny the vegetarian option
        last_intent = tracker.get_intent_of_latest_message()
        print(last_intent)
        if last_intent == 'denny':
            dispatcher.utter_message(text="Alright, besides that, do you have any ingredients you dont like?")
            return [SlotSet("prefer_vegetarian", False)]
        elif last_intent == 'affirm':
            dispatcher.utter_message(text="Veggie it is, besides that, do you have any ingredients you dont like?")
            return [SlotSet("prefer_vegetarian", True)]
        else:
            dispatcher.utter_message(text="I didn't get that, do you want a vegetarian recipe?")
            return []
        
class action_restart_ingredients(Action):
    def name(self) -> Text:
        return "action_restart_ingredients"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Alright, lets try it againg. What do we have in hand then?")
        return [SlotSet("ingredients_list", [])]
    

class ActionRecommendRecipe(Action):
    def name(self) -> Text:
        return "action_recommend_recipe"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # encode the query
        new_ingredients = tracker.get_slot('ingredients_list')
        excluded_ingredients = tracker.get_slot("excluded_ingredients") or []
        vegetarian = tracker.get_slot("prefer_vegetarian")
        # current_ingredients = tracker.get_slot("ingredients_list") or []
       

        # Check the ingredients to ensure there are no duplicates or sub-words
        filtered_ingredients = []
        for ingredient in new_ingredients:
            if ingredient not in filtered_ingredients:
                # Check if the ingredient is a sub-word of another ingredient
                is_sub_word = False
                for other_ingredient in new_ingredients:
                    if ingredient != other_ingredient and ingredient in other_ingredient:
                        is_sub_word = True
                        break
                if not is_sub_word:
                    filtered_ingredients.append(ingredient)

        # Update the prompt with the filtered ingredients
        if vegetarian:
            prompt = "A vegetarian recipe with these ingredients: " + ', '.join(filtered_ingredients)
        else:
            prompt = "A recipe with these ingredients: " + ', '.join(filtered_ingredients)
        
        print(prompt+'in process'+'/n')

        # prompt = "A recipe with these ingredients: " + ', '.join(ingredients)
        
        # Get the embeddings for the ingredients
        query_embedding = model.encode(prompt, convert_to_tensor=True)
        excluded_ingredients = tracker.get_slot("excluded_ingredients") or []
        if vegetarian:
            top_k = 2*50*len(excluded_ingredients) if excluded_ingredients else 100 
        else:
            top_k = 50*len(excluded_ingredients) if excluded_ingredients else 10 

        cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
       
        best_indexes = []
        best_titles = []

        for score, idx in zip(top_results[0], top_results[1]):
            idx = idx.item()  # Convert PyTorch tensor to a Python integer
            recipe = data.loc[idx]
            recipe_title = data.iloc[idx]['title']
            recipe_ingredients = recipe['NER'].split(', ')
            # Check if the recipe contains any excluded ingredients
            contains_excluded = False
            for ingredient in excluded_ingredients:
                if any(lemmatize_ingredient(ingredient) in lemmatize_ingredient(recipe_ingredient) for recipe_ingredient in recipe_ingredients):
                    print(f"Excluded ingredient found in recipe {recipe['title']}: {ingredient}", recipe_ingredients)
                    contains_excluded = True
                    break
            else:
                if vegetarian:
                    # Check in the dataset if is_vegetarian is True
                    if recipe['is_vegetarian']==True:
                        best_indexes.append(idx)
                    else:
                        print(f"Recipe {recipe['title']} is not vegetarian")
                        continue
        else:    
                    
                    best_indexes.append(idx)
        
        # best_indexes = [idx.item() for idx in top_results[1]]
        top3_indexes = best_indexes[:3]
        top3_recipe_titles = [data.iloc[idx]['title'] for idx in best_indexes[:3]]
        del best_indexes[:3]
        
        # Create buttons for the top 3 recipes
        buttons = [{"title": title, "payload": f"/choose_recipe{{\"selected_recipe_index\": {idx}}}"} 
        for title, idx in zip(top3_recipe_titles, top3_indexes)]
        buttons.append({"title": "None of these, show me more", "payload": "/more_options"})
        
        # Send a message with the buttons
        dispatcher.utter_message(text="Look!This are my top 3 choices for you, let me know if you want me to look for something different", buttons=buttons,)
        
        # Set the slot for the selected recipe indexes
        return [SlotSet("recommended_recipe_indexes", best_indexes)]


class ActionRecommendAnotherRecipe(Action):
    def name(self) -> Text:
        return "action_recommend_another_recipe"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get the selected recipe indexes
        best_indexes = tracker.get_slot('recommended_recipe_indexes')
        
        # Check if there are enough indexes
        if len(best_indexes) < 3:  # Adjusted condition
            dispatcher.utter_message(text="I need more ingredients to find a suitable recipe. Can you provide more?")
            return [SlotSet("recommended_recipe_indexes", None)]
        
        # Select 3 random indexes from the list
        random_indexes = random.sample(best_indexes, 3)
        for idx in random_indexes:
            best_indexes.remove(idx)    
        
        # Get titles of recipes corresponding to selected indexes
        recipe_titles = [data.iloc[idx]['title'] for idx in random_indexes]

        # Define buttons
        buttons = [{"title": title, "payload": f"/choose_recipe{{\"selected_recipe_index\": {idx}}}"} 
                   for title, idx in zip(recipe_titles, random_indexes)]
        buttons.append({"title": "None of these, show me more", "payload": "/more_options"})

        # Send a message with the recipe options
        dispatcher.utter_message(text="What about one of these others?", buttons=buttons)

        # Update recommended_recipe_indexes slot
        return [SlotSet("recommended_recipe_indexes", best_indexes)]
    
class ActionShowRecipe(Action):
    def name(self) -> Text:
        return "action_show_recipe"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get the selected recipe index
        recipe_index = tracker.get_slot('selected_recipe_index')
        
        # Get the recipe title
        recipe_title = data.iloc[recipe_index]['title']
        
        # Get the recipe ingredients
        recipe_ingredients = data.iloc[recipe_index]['ingredients']
        
        # Get the recipe instructions
        recipe_instructions = data.iloc[recipe_index]['directions']

        message = f"Great Choice! For the recipe '{recipe_title}', the ingredients are: {recipe_ingredients}. The instructions are: {recipe_instructions}. Is it okay for you?"
        
        # Send a message with the recipe
        dispatcher.utter_message(text=message)
        
        return []