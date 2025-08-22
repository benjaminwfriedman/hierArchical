from hierarchical.abstractions import Model

model = Model.from_ifc(ifc_file="/Users/benjaminfriedman/repos/hierArchical/sample_house.ifc")

# Example usage
# view the spaces + objects
model.show_spaces(by="omniclass_space_type")
model.show_objects()

# view the full building graph
model.show_building_graph()

model.building_graph.query_to_string("MATCH (d:Object {type: 'door'}) RETURN d.id AS Door_ID, d.width AS Width", return_type='dict')

# Example Ask
print("Q: What spaces are in the model?")
print("A:", model.ask("What spaces are in the model?"))