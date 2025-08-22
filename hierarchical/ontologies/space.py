from openai import OpenAI
from dotenv import load_dotenv
import os

from hierarchical.abstractions import Space

# Load environment variables from .env file
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

class Ontology:
    """Base class for ontologies"""
    
    def __init__(self):
        self.client = OpenAI()

    def get_openai_client(self):
        """Returns the OpenAI client instance"""
        client = OpenAI(api_key=OPEN_AI_API_KEY)
        return client

class OmniclassSpaceOntology(Ontology):
    """Class to hold the Omniclass Space Type ontology"""
    
    def __init__(self):

        self.client = self.get_openai_client()
        self.space_types = [
            # gathering spaces
            "briefing room",
            "seminar room", 
            "classroom",
            "lecture hall",
            "computer lab",
            "assembly hall",
            "information counter",
            "social room",
            "living room",
            "reception space",
            "other gathering spaces",
            
            # performance spaces
            "acting stage",
            "lectern",
            "orchestra pit",
            "choir loft",
            "performance rehearsal space",
            "sound stage",
            "production stage",
            "performance hall",
            "amphitheater",
            "concert hall",
            "auditorium",
            "theater",
            "other general performance spaces",
            "pre-function lobby",
            "seating section",
            "seating",
            "seating aisle",
            "bleacher",
            "viewing room",
            "projection booth",
            "catwalk",
            "stage wings",
            "stage side",
            "other supporting performance spaces",
            
            # food and beverage spaces
            "kitchen",
            "preparation",
            "cooking",
            "device cleaning",
            "other cooking spaces",
            "dining room",
            "banquet hall",
            "food court",
            "snack bar",
            "salad bar",
            "liquor bar",
            "beverage station",
            "table bussing station",
            "serving station",
            "dining hall",
            "cafeteria",
            "servery",
            "tray return space",
            "food discard station",
            "other dining and drinking spaces",
            
            # meeting spaces
            "meeting room",
            "council chambers",
            "conference room",
            "press conference room",
            "community room",
            "war room",
            "interrogation space",
            "interview room",
            "consultation room",
            "other meeting spaces",
            
            # creative, study, and administrative spaces
            "recording studio",
            "artist's studio",
            "audiovisual editing space",
            "printing room",
            "laboratory",
            "study room",
            "reading room",
            "library",
            "office",
            "office cubicle",
            "open office space",
            "mail room",
            "sorting room",
            "copy room",
            "court room",
            "jury box",
            "jury room",
            "judge's bench",
            "judge's chambers",
            "witness stand",
            "hearing room",
            "other creative, study, and administrative spaces",
            
            # production, fabrication, and maintenance spaces
            "manufacturing space",
            "clean room",
            "processing room",
            "material handling area",
            "batching space",
            "mixing space",
            "parts assembly space",
            "containment room",
            "product testing space",
            "product inspection space",
            "production observation space",
            
            # cultural spaces
            "marriage sanctuary",
            "hupa",
            "baptistery",
            "circumcision space",
            "cathedra",
            "throne",
            "other transformation spaces",
            "sacred gateway",
            "sacred pathway",
            "sacred station",
            "other procession spaces",
            "art gallery",
            "museum gallery",
            "exhibit gallery",
            "sculpture garden",
            "ornamental garden",
            "observation deck",
            "zen garden",
            "other contemplation spaces",
            "crypt",
            "burial chamber",
            "casket compartment",
            "coffin",
            "morgue compartment",
            "grave space",
            "other death spaces",
            "other cultural spaces",
            
            # protection spaces
            "park shelter",
            "entry porch",
            "covered walkway",
            "canopy",
            "shielded room",
            "containment room",
            "other spaces for protection from the elements",
            "safe room",
            "bunker",
            "bomb shelter",
            "other spaces for protection from violence",
            
            # securing spaces
            "cage",
            "animal stall",
            "stable",
            "kennel",
            "aquarium",
            "other animal securing spaces",
            "detention cell",
            "holding cell",
            "other detention spaces",
            
            # storage spaces
            "storage room",
            "closet",
            "coat check",
            "locker room",
            "filing space",
            "supply room",
            "warehouse space",
            "vehicular storage space",
            "garage",
            "parking lot",
            "waste storage space",
            "recycling storage space",
            "cupboard",
            "storage shelf",
            "drawer",
            "cubby compartment",
            "locker compartment",
            "fixed storage bin",
            "other fixed storage spaces",
            "vehicle storage compartment",
            "trunk",
            "glove box",
            "portable bin",
            "basket",
            "box",
            "vessel",
            "pot",
            "vase",
            "other non-fixed location storage spaces",
            "environmental room",
            "refrigeration compartment",
            "freezing compartment",
            "dry storage compartment",
            "humidity controlled storage space",
            "vacuum sealed storage compartment",
            "other environmentally controlled storage space",
            "sanitary storage room",
            "soiled storage room space",
            "sacristy",
            "vestry",
            "hazardous material storage space",
            "organic remains storage space",
            "book stacks",
            "baggage claim",
            "evidence room",
            "vehicle impound lot",
            "other special storage space",
            
            # facility service spaces
            "access chamber",
            "area way",
            "crawl space",
            "service space",
            "air shaft",
            "light well",
            "other general facility service spaces",
            "equipment room",
            "computer server room",
            "refrigerant machinery room",
            "furnace room",
            "incinerator room",
            "electrical room",
            "telecommunications room",
            "transformer vault",
            "elevator shaft",
            "dumbwaiter shaft",
            "other facility equipment service spaces",
            "electrical line",
            "electrical conduit chamber",
            "electrical duct bank",
            "other power distribution spaces",
            "communications line",
            "communications duct bank",
            "cable tray",
            "other information signal distribution spaces",
            "gas pipeline",
            "medical gas pipe",
            "vacuum pipe",
            "laboratory gas pipe",
            "air supply duct",
            "air return duct",
            "exhaust duct",
            "gas piping chase",
            "mechanical shaft",
            "other gas distribution spaces",
            "oil pipeline",
            "water pipeline",
            "chilled water pipe",
            "hot water pipe",
            "special water pipe",
            "liquid pipe chase",
            "other liquid distribution spaces",
            "other service distribution spaces",
            "other facility service spaces",
            
            # circulation spaces
            "corridor",
            "hallway",
            "aisle",
            "mall",
            "concourse",
            "atrium",
            "breezeway",
            "jet way",
            "moving walkway",
            "other horizontal circulation spaces",
            "stairway",
            "egress stairway",
            "ceremonial stairway",
            "monumental stairway",
            "escalator",
            "ramp",
            "stair and ramp combination",
            "elevator cab",
            "dumbwaiter",
            "other vertical circulation spaces",
            "entry vestibule",
            "entry lobby",
            "elevator lobby",
            "landing",
            "anteroom",
            "air lock",
            "pressure lock",
            "other transitional circulation spaces",
            "means of egress",
            "accessible route",
            "hub room",
            "other specialty circulation spaces",
            
            # travel spaces
            "highway",
            "causeway",
            "street",
            "alley",
            "driveway",
            "drop-off area",
            "loading dock",
            "entrance/exit ramp",
            "bridge",
            "airport apron",
            "taxiway",
            "runway",
            "airway",
            "waterway",
            "channel",
            "canal",
            "bay",
            "dock",
            "pier",
            "slip",
            "other vehicular travel spaces",
            "sidewalk",
            "pedestrian way",
            "footpath",
            "trail",
            "gangway",
            "other pedestrian travel spaces"
        ]

    def apply_ontology(self, space: Space):
        """Applies the Omniclass Space Ontology to a given space."""
        
        # Gather context about the space for classification
        space_context = {
            "name": space.name,
            "attributes": dict(space.attributes) if space.attributes else {},
            # "ontologies": dict(space.ontologies) if space.ontologies else {}
        }
        
        # Create a prompt for the OpenAI API to classify the space
        prompt = f"""
        You are an expert in architectural space classification using the Omniclass Space Type ontology.
        
        Given the following space information:
        - Name: {space_context['name']}
        - Current Attributes: {space_context['attributes']}
        
        Please classify this space according to the Omniclass Space Type ontology. 
        Choose the MOST APPROPRIATE space type from this list:
        {', '.join(self.space_types)}
        
        Respond with ONLY the exact space type name from the list above that best matches this space.
        If no perfect match exists, choose the closest appropriate category.

        all answers should be lowercase
        """
        
        try:
            # Use OpenAI to classify the space
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert architectural space classifier. Respond only with the exact space type name from the provided Omniclass ontology list."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1  # Low temperature for consistent classification
            )
            
            classified_space_type = response.choices[0].message.content.strip()
            
            # Add the classification to the space's attributes
            space.attributes['omniclass_space_type'] = classified_space_type
            ontology_attrs = {
                "omniclass_space_type": classified_space_type
            }
           
            
            return space, ontology_attrs
        except Exception as e:
            print(f"Error classifying space '{space.name}': {e}")
