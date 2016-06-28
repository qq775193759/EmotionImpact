import xml.etree.ElementTree as ET

tree = ET.parse('ACCEDEdescription.xml')
root = tree.getroot()

l = []
for child in root:
    d = {}
    d['id'] = child[0].text
    d['name'] = child[1].text
    d['movie'] = child[3].text
    l.append(d)


print(d)