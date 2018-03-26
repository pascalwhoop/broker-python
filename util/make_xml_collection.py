"""
Module to create a collection of unique xml samples by letting a broker run through all its received xml messages and
create unions of the xml trees of each message type
"""
from bs4 import BeautifulSoup, NavigableString
import pickle


def parse_message(xml:str):
    """Handles the message (which is an xml string) and parses it. Unknown tags get added to the dict of tags and known
    tags are used to expand the previously seen ones in case they hold new types

    :xml:str: TODO
    :returns: TODO

    """
    return BeautifulSoup(xml, "xml")


# holds our known types. It holds the *root* BeautifulSoup object, not the direct descendant. So to get the
# descendant it's usually necessary to do: xml_types['some-type'].findChild().something
xml_types = {}


def merge_xml(source: BeautifulSoup, target: BeautifulSoup):
    """this function recursively merges a node into a target node. Attributes are overwritten if present in both.
    If source has a child that target doesn't have, it's added. If it already has the same child, it's merged recursively.
    Multiple children """

    # adding all the nodes to the target
    for child in source.children:
        #skipping those, not useful
        if type(child) is NavigableString:
            continue
        child_name = child.name

        target_children_names = [tc.name for tc in list(target.children)]
        if child_name in target_children_names:
            merge_xml(child, target.find(child_name))
            continue
        else:
            target.append(child)

    # adding all the attributes(keys) of the node to the target node
    for attr in source.attrs:
        if attr not in target.attrs:
            target[attr] = source[attr]

def add_to_type_set(xml_obj: BeautifulSoup):
    """expects bf object"""
    tag_name = xml_obj.findChild().name
    global xml_types
    if tag_name not in xml_types:
        xml_types[tag_name] = xml_obj
    else:
        merge_xml(xml_obj, xml_types[tag_name])


def pickle_xml():
    """saving to disk do make sure we have the xml types
    :returns: TODO

    """
    for x in xml_types:
        fn = "tests/xml_msgs/{}.xml".format(x)
        with open(fn, "w") as f:
            f.write(xml_types[x].prettify())
