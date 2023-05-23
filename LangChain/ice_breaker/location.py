import geocoder 
import googlemaps


def location_finder(text):

    # Get the location information 
    g = geocoder.ip('me') 

    # Get the location of the user 
    location = g.latlng  


    # API key
    api_key = 'AIzaSyA1WLfZuDVgGp12qieXj89_9J_Tg0T_Y48'

    # Create a gmaps object
    gmaps = googlemaps.Client(key=api_key)
    result_list = []


    # Search for nearby pet hospitals
    nearby_pet_hospitals = gmaps.places_nearby(location=(location[0], location[1]), 
                                            keyword=text,
                                            radius = 2000)

    # Print the names of the pet hospitals
    for place in nearby_pet_hospitals['results']:
        result_list.append(place['name'])
    
    return result_list



print(location_finder("Pet hospitals near me"))