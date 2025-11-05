"""
XBRL Taxonomy to RDF Ontology Converter
Converts IFRS SDS XBRL taxonomy files to RDF/OWL ontology
"""

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, SKOS, DCTERMS

# === CONFIGURATION ===
IFRS = Namespace("https://xbrl.ifrs.org/sds#")
XBRL = Namespace("http://xbrl.org/")
XLINK = "{http://www.w3.org/1999/xlink}"
LINK = "{http://www.xbrl.org/2003/linkbase}"
ARCROLE = Namespace("http://xbrl.org/int/arcrole/")
XLINK_NS = "http://www.w3.org/1999/xlink"


def extract_loc_map(tree):
    # Extract locators (xlink:label -> URI)
    loc_map = {}
    for loc in tree.findall(f".//{LINK}loc"):
        label = loc.attrib.get(f"{XLINK}label")
        href = loc.attrib.get(f"{XLINK}href")
        if href and label:
            # Normalize href: remove "../../" and base to IFRS URI
            concept = href.split("#")[-1]
            loc_map[label] = f"https://xbrl.ifrs.org/sds#{concept}"
    return loc_map

def parse_taxonomy_folder(g, folder_path):
    print("Parsing role definitions ...")
    # Parse role definitions (.xsd file)
    for file in os.listdir(folder_path):
        if file.startswith("rol_") and file.endswith(".xsd"):
            tree = ET.parse(os.path.join(folder_path, file))
            for role in tree.findall(f".//{LINK}roleType"):
                role_id = role.attrib.get("id")
                role_uri = role.attrib.get("roleURI")
                definition = role.findtext(f"{LINK}definition")
                if role_uri:
                    g.add((URIRef(role_uri), RDF.type, SKOS.Collection))
                    if definition:
                        g.add((URIRef(role_uri), SKOS.prefLabel, Literal(definition)))

    print("Parsing presentation linkbases ...")
    # Parse presentation linkbases (pre_ifrs...xml files)
    for file in os.listdir(folder_path):
        if file.startswith("pre_") and file.endswith(".xml"):
            tree = ET.parse(os.path.join(folder_path, file))
            loc_map = extract_loc_map(tree)
            for arc in tree.findall(f".//{LINK}presentationArc"):
                src = arc.attrib.get(f"{XLINK}from")
                dst = arc.attrib.get(f"{XLINK}to")
                if src in loc_map and dst in loc_map:
                    g.add((URIRef(loc_map[dst]), SKOS.broader, URIRef(loc_map[src])))

    print("Parsing definition linkbases ...")
    # Parse definition linkbases (def_ifrs...xml files)
    for file in os.listdir(folder_path):
        if file.startswith("def_") and file.endswith(".xml"):
            tree = ET.parse(os.path.join(folder_path, file))
            loc_map = extract_loc_map(tree)
            for arc in tree.findall(f".//{LINK}definitionArc"):
                src = arc.attrib.get(f"{XLINK}from")
                dst = arc.attrib.get(f"{XLINK}to")
                arcrole = arc.attrib.get(f"{XLINK}arcrole", "")
                if src in loc_map and dst in loc_map:
                    if "domain-member" in arcrole:
                        g.add((URIRef(loc_map[dst]), IFRS.hasMember, URIRef(loc_map[src])))
                    elif "dimension-domain" in arcrole:
                        g.add((URIRef(loc_map[dst]), IFRS.hasDomain, URIRef(loc_map[src])))
                    else:
                        g.add((URIRef(loc_map[dst]), RDFS.seeAlso, URIRef(loc_map[src])))

    print("Parsing reference linkbases ...")
    # Parse reference linkbases (ref_ifrs...xml files)
    for file in os.listdir(folder_path):
        if file.startswith("ref_") and file.endswith(".xml"):
            tree = ET.parse(os.path.join(folder_path, file))
            loc_map = extract_loc_map(tree)
            for arc in tree.findall(f".//{LINK}referenceArc"):
                src = arc.attrib.get(f"{XLINK}from")
                dst = arc.attrib.get(f"{XLINK}to")
                if src in loc_map:
                    g.add((URIRef(loc_map[src]), DCTERMS.source, Literal(dst)))
            # Add IFRS reference info (e.g., IFRS, S2, paragraph)
            for ref in tree.findall(f".//{LINK}reference"):
                label = ref.attrib.get(f"{XLINK}label")
                name = ref.findtext("{http://www.xbrl.org/2003/reference}Name")
                number = ref.findtext("{http://www.xbrl.org/2003/reference}Number")
                paragraph = ref.findtext("{http://www.xbrl.org/2003/reference}Paragraph")
                if label and name and number and paragraph:
                    ref_uri = URIRef(f"https://xbrl.ifrs.org/sds/ref/{name}-{number}-{paragraph}")
                    g.add((ref_uri, RDF.type, DCTERMS.BibliographicResource))
                    g.add((ref_uri, DCTERMS.identifier, Literal(f"{name} {number} ¶{paragraph}")))

def parse_labels(g, labels_dir):
    print("Collecting label resources ...")
    # Collect label resources (id -> text, lang, role)
    label_resources = {}
    for file in os.listdir(labels_dir):
        if file.startswith(("lab_", "doc_")) and file.endswith(".xml"):
            tree = ET.parse(os.path.join(labels_dir, file))
            for lbl in tree.findall(f".//{LINK}label"):
                label_id = lbl.attrib.get("id") or lbl.attrib.get(f"{XLINK}label")
                role = lbl.attrib.get(f"{XLINK}role", "")
                lang = lbl.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "en")
                text = lbl.text.strip() if lbl.text else ""
                if label_id and text:
                    label_resources[label_id] = {
                        "text": text,
                        "lang": lang,
                        "role": role
                    }

    print("Mapping concepts to labels ...")
    # Map concept (loc) to label (res)
    for file in os.listdir(labels_dir):
        if file.startswith("in_") and file.endswith(".xml"):
            tree = ET.parse(os.path.join(labels_dir, file))
            loc_map = extract_loc_map(tree)
            for arc in tree.findall(f".//{LINK}labelArc"):
                src = arc.attrib.get(f"{XLINK}from")  # loc_#
                dst = arc.attrib.get(f"{XLINK}to")    # res_#
                if src in loc_map and dst in label_resources:
                    concept_uri = URIRef(loc_map[src])
                    lbl = label_resources[dst]
                    text = Literal(lbl["text"], lang=lbl["lang"])

                    # Determine label type from role
                    if "documentation" in lbl["role"]:
                        g.add((concept_uri, SKOS.definition, text))
                    elif "label" in lbl["role"] or "terseLabel" in lbl["role"]:
                        g.add((concept_uri, SKOS.prefLabel, text))
                    elif "measurementGuidance" in lbl["role"]:
                        g.add((concept_uri, RDFS.comment, text))
                    else:
                        g.add((concept_uri, SKOS.altLabel, text))


class XBRLToRDFConverter:
    """Converts XBRL taxonomy to RDF ontology
    
    This converter processes XBRL taxonomy files and generates an RDF/OWL ontology
    that represents the structure, concepts, and relationships defined in the taxonomy.
    
    Relationship Types Extracted:
    
    1. PRESENTATION RELATIONSHIPS (from presentation linkbase):
       - hasParent/hasChild: Hierarchical parent-child relationships
       - order: Sequence order in presentation
       Source: pre_*.xml files
       
    2. DIMENSIONAL RELATIONSHIPS (from definition linkbase):
       - hasDimension: Links hypercube to dimensions
       - hasDomain: Links dimension to its domain
       - hasMember: Links domain to its members
       - hasHypercube: Links primary item to hypercube (via 'all' arcrole)
       - isClosed: Boolean indicating if dimension is closed
       Source: def_*.xml files
       Arcroles:
         - http://xbrl.org/int/dim/arcrole/hypercube-dimension
         - http://xbrl.org/int/dim/arcrole/dimension-domain
         - http://xbrl.org/int/dim/arcrole/domain-member
         - http://xbrl.org/int/dim/arcrole/all
    
    3. ROLE-ELEMENT RELATIONSHIPS (from generic reference linkbase):
       - referencesElement: Links roles to concepts they reference
       Source: gre_*.xml files (in linkbases/ifrs_s1/, linkbases/ifrs_s2/, dimensions/)
       Arcrole: http://xbrl.org/arcrole/2008/element-reference
    
    4. CONCEPT-REFERENCE RELATIONSHIPS (from reference linkbase):
       - hasReference: Indicates concept has authoritative references
       Source: ref_*.xml files
    
    5. LABEL RELATIONSHIPS (from label linkbases):
       - rdfs:label: Standard labels
       - documentation: Documentation text
       - inlineLabel: Inline XBRL specific labels
       Source: lab_*.xml, doc_*.xml, in_*.xml, gla_*.xml files
    
    6. EXPLICIT DIMENSION RELATIONSHIPS (from dimensions/ folder):
       - Additional dimensional relationships specific to dimension definitions
       - Enhanced dimension-domain and domain-member relationships
       Source: dimensions/dim_*.xml files
       Note: The dimensions/ folder may contain more detailed or specialized
             dimensional relationships compared to the standard definition linkbases
    """
    
    # XML Namespaces
    NAMESPACES = {
        'xsd': 'http://www.w3.org/2001/XMLSchema',
        'link': 'http://www.xbrl.org/2003/linkbase',
        'xlink': 'http://www.w3.org/1999/xlink',
        'xbrldt': 'http://xbrl.org/2005/xbrldt',
        'gen': 'http://xbrl.org/2008/generic',
        'label': 'http://xbrl.org/2008/label',
        'ref': 'http://www.xbrl.org/2003/role/reference'
    }
    
    def __init__(self, taxonomy_path, output_file='ifrs_sds_ontology.rdf'):
        self.taxonomy_path = Path(taxonomy_path)
        self.output_file = output_file
        
        # Data structures
        self.concepts = {}
        self.roles = {}
        self.presentation_arcs = []
        self.definition_arcs = []
        self.dimension_arcs = []  # Additional dimension relationships from dimensions/ folder
        self.labels = {}
        self.documentation = {}
        self.references = {}
        self.role_references = {}  # Role to element references (from gre files)
        self.inline_labels = {}  # Additional inline labels (from in files)
        
        # Register namespaces for parsing
        for prefix, uri in self.NAMESPACES.items():
            ET.register_namespace(prefix, uri)
    
    def parse_schema_file(self, schema_file):
        """Parse XSD schema file to extract concepts"""
        print(f"Parsing schema: {schema_file}")
        tree = ET.parse(schema_file)
        root = tree.getroot()
        
        # Extract elements (concepts)
        for elem in root.findall('.//xsd:element', self.NAMESPACES):
            name = elem.get('name')
            if name:
                concept_type = elem.get('type', '')
                substitution_group = elem.get('substitutionGroup', '')
                abstract = elem.get('abstract', 'false')
                
                self.concepts[name] = {
                    'name': name,
                    'type': concept_type,
                    'substitutionGroup': substitution_group,
                    'abstract': abstract == 'true',
                    'uri': f"https://xbrl.ifrs.org/ontology/ifrs-sds/{name}"
                }
    
    def parse_role_file(self, role_file):
        """Parse role definition file"""
        print(f"Parsing roles: {role_file}")
        tree = ET.parse(role_file)
        root = tree.getroot()
        
        # Extract roleType definitions
        for role_type in root.findall('.//link:roleType', self.NAMESPACES):
            role_uri = role_type.get('roleURI')
            role_id = role_type.get('id')
            
            definition_elem = role_type.find('link:definition', self.NAMESPACES)
            definition = definition_elem.text if definition_elem is not None else ''
            
            if role_uri:
                self.roles[role_id] = {
                    'uri': role_uri,
                    'id': role_id,
                    'definition': definition
                }
    
    def parse_presentation_linkbase(self, pres_file):
        """Parse presentation linkbase for hierarchical relationships"""
        print(f"Parsing presentation: {pres_file}")
        tree = ET.parse(pres_file)
        root = tree.getroot()
        
        # Build locator map
        locators = {}
        for loc in root.findall('.//link:loc', self.NAMESPACES):
            label = loc.get('{http://www.w3.org/1999/xlink}label')
            href = loc.get('{http://www.w3.org/1999/xlink}href')
            if label and href:
                # Extract concept name from href
                concept_name = self.extract_concept_from_href(href)
                locators[label] = concept_name
        
        # Extract presentation arcs
        for arc in root.findall('.//link:presentationArc', self.NAMESPACES):
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            to_label = arc.get('{http://www.w3.org/1999/xlink}to')
            order = arc.get('order', '0')
            
            if from_label in locators and to_label in locators:
                self.presentation_arcs.append({
                    'parent': locators[from_label],
                    'child': locators[to_label],
                    'order': float(order)
                })
    
    def parse_definition_linkbase(self, def_file):
        """Parse definition linkbase for dimensional relationships"""
        print(f"Parsing definition: {def_file}")
        tree = ET.parse(def_file)
        root = tree.getroot()
        
        # Build locator map
        locators = {}
        for loc in root.findall('.//link:loc', self.NAMESPACES):
            label = loc.get('{http://www.w3.org/1999/xlink}label')
            href = loc.get('{http://www.w3.org/1999/xlink}href')
            if label and href:
                concept_name = self.extract_concept_from_href(href)
                locators[label] = concept_name
        
        # Extract definition arcs
        for arc in root.findall('.//link:definitionArc', self.NAMESPACES):
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            to_label = arc.get('{http://www.w3.org/1999/xlink}to')
            arcrole = arc.get('{http://www.w3.org/1999/xlink}arcrole', '')
            closed = arc.get('{http://xbrl.org/2005/xbrldt}closed', 'false')
            
            if from_label in locators and to_label in locators:
                self.definition_arcs.append({
                    'from': locators[from_label],
                    'to': locators[to_label],
                    'arcrole': arcrole,
                    'closed': closed == 'true'
                })
    
    def parse_label_linkbase(self, label_file):
        """Parse label linkbase for human-readable labels"""
        print(f"Parsing labels: {label_file}")
        tree = ET.parse(label_file)
        root = tree.getroot()
        
        # Build locator map
        locators = {}
        for loc in root.findall('.//link:loc', self.NAMESPACES):
            label = loc.get('{http://www.w3.org/1999/xlink}label')
            href = loc.get('{http://www.w3.org/1999/xlink}href')
            if label and href:
                concept_name = self.extract_concept_from_href(href)
                locators[label] = concept_name
        
        # Map labels to concepts via arcs
        label_map = {}
        for arc in root.findall('.//link:labelArc', self.NAMESPACES):
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            to_label = arc.get('{http://www.w3.org/1999/xlink}to')
            if from_label in locators:
                label_map[to_label] = locators[from_label]
        
        # Extract label resources
        for label_elem in root.findall('.//link:label', self.NAMESPACES):
            label_id = label_elem.get('{http://www.w3.org/1999/xlink}label')
            role = label_elem.get('{http://www.w3.org/1999/xlink}role', '')
            lang = label_elem.get('{http://www.w3.org/XML/1998/namespace}lang', 'en')
            text = label_elem.text or ''
            
            if label_id in label_map:
                concept_name = label_map[label_id]
                
                if 'documentation' in role:
                    if concept_name not in self.documentation:
                        self.documentation[concept_name] = {}
                    self.documentation[concept_name][lang] = text
                else:
                    if concept_name not in self.labels:
                        self.labels[concept_name] = {}
                    self.labels[concept_name][lang] = text
    
    def parse_reference_linkbase(self, ref_file):
        """Parse reference linkbase for authoritative references"""
        print(f"Parsing references: {ref_file}")
        tree = ET.parse(ref_file)
        root = tree.getroot()
        
        # Build locator map
        locators = {}
        for loc in root.findall('.//link:loc', self.NAMESPACES):
            label = loc.get('{http://www.w3.org/1999/xlink}label')
            href = loc.get('{http://www.w3.org/1999/xlink}href')
            if label and href:
                concept_name = self.extract_concept_from_href(href)
                locators[label] = concept_name
        
        # Extract reference arcs (simplified - just mark that references exist)
        for arc in root.findall('.//link:referenceArc', self.NAMESPACES):
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            if from_label in locators:
                concept_name = locators[from_label]
                self.references[concept_name] = True
    
    def parse_generic_reference_linkbase(self, gre_file):
        """Parse generic reference linkbase (gre files) - links roles to elements"""
        print(f"Parsing generic references: {gre_file}")
        tree = ET.parse(gre_file)
        root = tree.getroot()
        
        # Build locator map - these can point to roles or concepts
        locators = {}
        for loc in root.findall('.//link:loc', self.NAMESPACES):
            label = loc.get('{http://www.w3.org/1999/xlink}label')
            href = loc.get('{http://www.w3.org/1999/xlink}href')
            if label and href:
                # Check if it's a role reference or concept reference
                if 'rol_' in href or '#ifrs_s' in href:
                    # Role reference
                    locators[label] = {'type': 'role', 'ref': href}
                else:
                    # Concept reference
                    concept_name = self.extract_concept_from_href(href)
                    if concept_name:
                        locators[label] = {'type': 'concept', 'ref': concept_name}
        
        # Extract generic arcs - element-reference arcrole
        for arc in root.findall('.//gen:arc', self.NAMESPACES):
            arcrole = arc.get('{http://www.w3.org/1999/xlink}arcrole', '')
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            to_label = arc.get('{http://www.w3.org/1999/xlink}to')
            
            if 'element-reference' in arcrole and from_label in locators:
                loc_data = locators[from_label]
                if loc_data['type'] == 'role':
                    # Store that this role references certain elements
                    role_ref = loc_data['ref']
                    if role_ref not in self.role_references:
                        self.role_references[role_ref] = []
                    self.role_references[role_ref].append({
                        'arcrole': arcrole,
                        'from': from_label,
                        'to': to_label
                    })
    
    def parse_inline_label_linkbase(self, in_file):
        """Parse inline label linkbase (in files) - additional labels for concepts"""
        print(f"Parsing inline labels: {in_file}")
        tree = ET.parse(in_file)
        root = tree.getroot()
        
        # Build locator map
        locators = {}
        for loc in root.findall('.//link:loc', self.NAMESPACES):
            label = loc.get('{http://www.w3.org/1999/xlink}label')
            href = loc.get('{http://www.w3.org/1999/xlink}href')
            if label and href:
                concept_name = self.extract_concept_from_href(href)
                locators[label] = concept_name
        
        # Map labels to concepts via arcs
        label_map = {}
        for arc in root.findall('.//link:labelArc', self.NAMESPACES):
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            to_label = arc.get('{http://www.w3.org/1999/xlink}to')
            if from_label in locators:
                label_map[to_label] = locators[from_label]
        
        # Extract label resources
        for label_elem in root.findall('.//link:label', self.NAMESPACES):
            label_id = label_elem.get('{http://www.w3.org/1999/xlink}label')
            role = label_elem.get('{http://www.w3.org/1999/xlink}role', '')
            lang = label_elem.get('{http://www.w3.org/XML/1998/namespace}lang', 'en')
            text = label_elem.text or ''
            
            if label_id in label_map:
                concept_name = label_map[label_id]
                
                # Store inline labels separately from main labels
                if concept_name not in self.inline_labels:
                    self.inline_labels[concept_name] = {}
                
                # Categorize by role
                role_type = 'inline'
                if 'documentation' in role:
                    role_type = 'inline_doc'
                elif 'label' in role:
                    role_type = 'inline_label'
                
                if role_type not in self.inline_labels[concept_name]:
                    self.inline_labels[concept_name][role_type] = {}
                
                self.inline_labels[concept_name][role_type][lang] = text
    
    def parse_dimension_linkbase(self, dim_file):
        """Parse dimension linkbase (dim files) - explicit dimension relationships"""
        print(f"Parsing dimensions: {dim_file}")
        tree = ET.parse(dim_file)
        root = tree.getroot()
        
        # Build locator map
        locators = {}
        for loc in root.findall('.//link:loc', self.NAMESPACES):
            label = loc.get('{http://www.w3.org/1999/xlink}label')
            href = loc.get('{http://www.w3.org/1999/xlink}href')
            if label and href:
                concept_name = self.extract_concept_from_href(href)
                locators[label] = concept_name
        
        # Extract dimension arcs - similar to definition arcs but specifically for dimensions
        for arc in root.findall('.//link:definitionArc', self.NAMESPACES):
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            to_label = arc.get('{http://www.w3.org/1999/xlink}to')
            arcrole = arc.get('{http://www.w3.org/1999/xlink}arcrole', '')
            closed = arc.get('{http://xbrl.org/2005/xbrldt}closed', 'false')
            order = arc.get('order', '0')
            
            if from_label in locators and to_label in locators:
                self.dimension_arcs.append({
                    'from': locators[from_label],
                    'to': locators[to_label],
                    'arcrole': arcrole,
                    'closed': closed == 'true',
                    'order': float(order)
                })
    
    def extract_concept_from_href(self, href):
        """Extract concept name from XBRL href attribute"""
        # href format: "../../ifrs_sds-cor_2024-04-26.xsd#ifrs-sds_ConceptName"
        if '#' in href:
            parts = href.split('#')
            if len(parts) > 1:
                # Remove namespace prefix if present
                concept_id = parts[1]
                if '_' in concept_id:
                    return concept_id.split('_', 1)[1]
                return concept_id
        return None
    
    def determine_concept_class(self, concept):
        """Determine RDF class for a concept based on its properties"""
        name = concept['name']
        subst_group = concept.get('substitutionGroup', '')
        
        if concept.get('abstract'):
            return 'Abstract'
        elif 'Domain' in name:
            return 'Domain'
        elif 'Member' in name or 'Axis' in subst_group:
            return 'Member'
        elif 'Axis' in name:
            return 'Dimension'
        elif 'LineItems' in name or 'Table' in name:
            return 'LineItem'
        else:
            return 'LineItem'
    
    def generate_rdf_ontology(self):
        """Generate RDF ontology from parsed data"""
        print("\nGenerating RDF ontology...")
        
        rdf_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<rdf:RDF',
            '    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            '    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
            '    xmlns:owl="http://www.w3.org/2002/07/owl#"',
            '    xmlns:xsd="http://www.w3.org/2001/XMLSchema#"',
            '    xmlns:skos="http://www.w3.org/2004/02/skos/core#"',
            '    xmlns:dcterms="http://purl.org/dc/terms/"',
            '    xmlns:ifrs-sds="https://xbrl.ifrs.org/ontology/ifrs-sds/"',
            '    xmlns:xbrl="http://www.xbrl.org/2003/instance#"',
            '    xmlns:xbrldt="http://xbrl.org/2005/xbrldt#"',
            '    xml:base="https://xbrl.ifrs.org/ontology/ifrs-sds/">',
            '',
            '    <!-- Ontology Metadata -->',
            '    <owl:Ontology rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/">',
            '        <dcterms:title xml:lang="en">IFRS Sustainability Disclosure Standards (SDS) Ontology</dcterms:title>',
            '        <dcterms:description xml:lang="en">RDF ontology derived from IFRS Sustainability Disclosure Standards XBRL Taxonomy</dcterms:description>',
            f'        <dcterms:created rdf:datatype="http://www.w3.org/2001/XMLSchema#date">{datetime.now().strftime("%Y-%m-%d")}</dcterms:created>',
            '        <dcterms:publisher>IFRS Foundation</dcterms:publisher>',
            '        <owl:versionInfo>Generated from XBRL Taxonomy</owl:versionInfo>',
            '    </owl:Ontology>',
            ''
        ]
        
        # Add class definitions
        rdf_lines.extend(self.generate_class_definitions())
        
        # Add property definitions
        rdf_lines.extend(self.generate_property_definitions())
        
        # Add role instances
        rdf_lines.extend(self.generate_role_instances())
        
        # Add concept instances
        rdf_lines.extend(self.generate_concept_instances())
        
        rdf_lines.append('</rdf:RDF>')
        
        return '\n'.join(rdf_lines)
    
    def generate_class_definitions(self):
        """Generate OWL class definitions"""
        return [
            '    <!-- Core Classes -->',
            '    ',
            '    <owl:Class rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept">',
            '        <rdfs:label xml:lang="en">XBRL Concept</rdfs:label>',
            '        <rdfs:comment xml:lang="en">A concept defined in the IFRS SDS taxonomy</rdfs:comment>',
            '    </owl:Class>',
            '    ',
            '    <owl:Class rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/Role">',
            '        <rdfs:label xml:lang="en">XBRL Role</rdfs:label>',
            '        <rdfs:comment xml:lang="en">A role defining a disclosure table or presentation</rdfs:comment>',
            '    </owl:Class>',
            '    ',
            '    <owl:Class rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/Dimension">',
            '        <rdfs:label xml:lang="en">Dimension</rdfs:label>',
            '        <rdfs:subClassOf rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept"/>',
            '    </owl:Class>',
            '    ',
            '    <owl:Class rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/Domain">',
            '        <rdfs:label xml:lang="en">Domain</rdfs:label>',
            '        <rdfs:subClassOf rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept"/>',
            '    </owl:Class>',
            '    ',
            '    <owl:Class rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/Member">',
            '        <rdfs:label xml:lang="en">Domain Member</rdfs:label>',
            '        <rdfs:subClassOf rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept"/>',
            '    </owl:Class>',
            '    ',
            '    <owl:Class rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/LineItem">',
            '        <rdfs:label xml:lang="en">Line Item</rdfs:label>',
            '        <rdfs:subClassOf rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept"/>',
            '    </owl:Class>',
            '    ',
            '    <owl:Class rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/Abstract">',
            '        <rdfs:label xml:lang="en">Abstract Concept</rdfs:label>',
            '        <rdfs:subClassOf rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept"/>',
            '    </owl:Class>',
            '    '
        ]
    
    def generate_property_definitions(self):
        """Generate OWL property definitions"""
        return [
            '    <!-- Object Properties - Presentation Relationships -->',
            '    ',
            '    <owl:ObjectProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/hasParent">',
            '        <rdfs:label xml:lang="en">has parent</rdfs:label>',
            '        <rdfs:comment xml:lang="en">Presentation parent-child relationship from XBRL presentation linkbase</rdfs:comment>',
            '        <rdfs:domain rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept"/>',
            '        <rdfs:range rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept"/>',
            '    </owl:ObjectProperty>',
            '    ',
            '    <owl:ObjectProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/hasChild">',
            '        <rdfs:label xml:lang="en">has child</rdfs:label>',
            '        <rdfs:comment xml:lang="en">Inverse of hasParent relationship</rdfs:comment>',
            '        <owl:inverseOf rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/hasParent"/>',
            '    </owl:ObjectProperty>',
            '    ',
            '    <!-- Object Properties - Dimensional Relationships -->',
            '    ',
            '    <owl:ObjectProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/hasDimension">',
            '        <rdfs:label xml:lang="en">has dimension</rdfs:label>',
            '        <rdfs:comment xml:lang="en">Links a concept to its dimensional structure (hypercube-dimension)</rdfs:comment>',
            '        <rdfs:range rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Dimension"/>',
            '    </owl:ObjectProperty>',
            '    ',
            '    <owl:ObjectProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/hasDomain">',
            '        <rdfs:label xml:lang="en">has domain</rdfs:label>',
            '        <rdfs:comment xml:lang="en">Links a dimension to its domain (dimension-domain)</rdfs:comment>',
            '        <rdfs:domain rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Dimension"/>',
            '        <rdfs:range rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Domain"/>',
            '    </owl:ObjectProperty>',
            '    ',
            '    <owl:ObjectProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/hasMember">',
            '        <rdfs:label xml:lang="en">has member</rdfs:label>',
            '        <rdfs:comment xml:lang="en">Links a domain to its members (domain-member)</rdfs:comment>',
            '        <rdfs:domain rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Domain"/>',
            '        <rdfs:range rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Member"/>',
            '    </owl:ObjectProperty>',
            '    ',
            '    <owl:ObjectProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/hasHypercube">',
            '        <rdfs:label xml:lang="en">has hypercube</rdfs:label>',
            '        <rdfs:comment xml:lang="en">Links a concept to a hypercube that defines its dimensional structure (all relationship)</rdfs:comment>',
            '        <rdfs:domain rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept"/>',
            '    </owl:ObjectProperty>',
            '    ',
            '    <!-- Object Properties - Role Relationships -->',
            '    ',
            '    <owl:ObjectProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/usedInRole">',
            '        <rdfs:label xml:lang="en">used in role</rdfs:label>',
            '        <rdfs:comment xml:lang="en">Associates a concept with a role in which it is used</rdfs:comment>',
            '        <rdfs:domain rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept"/>',
            '        <rdfs:range rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Role"/>',
            '    </owl:ObjectProperty>',
            '    ',
            '    <owl:ObjectProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/referencesElement">',
            '        <rdfs:label xml:lang="en">references element</rdfs:label>',
            '        <rdfs:comment xml:lang="en">Links a role to elements it references (from generic reference linkbase)</rdfs:comment>',
            '        <rdfs:domain rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Role"/>',
            '        <rdfs:range rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept"/>',
            '    </owl:ObjectProperty>',
            '    ',
            '    <!-- Object Properties - Reference Relationships -->',
            '    ',
            '    <owl:ObjectProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/hasReference">',
            '        <rdfs:label xml:lang="en">has reference</rdfs:label>',
            '        <rdfs:comment xml:lang="en">Links a concept to its authoritative reference in IFRS standards</rdfs:comment>',
            '        <rdfs:domain rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Concept"/>',
            '    </owl:ObjectProperty>',
            '    ',
            '    <!-- Datatype Properties -->',
            '    ',
            '    <owl:DatatypeProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/conceptName">',
            '        <rdfs:label xml:lang="en">concept name</rdfs:label>',
            '        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>',
            '    </owl:DatatypeProperty>',
            '    ',
            '    <owl:DatatypeProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/roleURI">',
            '        <rdfs:label xml:lang="en">role URI</rdfs:label>',
            '        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#anyURI"/>',
            '    </owl:DatatypeProperty>',
            '    ',
            '    <owl:DatatypeProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/roleDefinition">',
            '        <rdfs:label xml:lang="en">role definition</rdfs:label>',
            '        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>',
            '    </owl:DatatypeProperty>',
            '    ',
            '    <owl:DatatypeProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/documentation">',
            '        <rdfs:label xml:lang="en">documentation</rdfs:label>',
            '        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>',
            '    </owl:DatatypeProperty>',
            '    ',
            '    <owl:DatatypeProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/inlineLabel">',
            '        <rdfs:label xml:lang="en">inline label</rdfs:label>',
            '        <rdfs:comment xml:lang="en">Additional inline label from inline label linkbase</rdfs:comment>',
            '        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>',
            '    </owl:DatatypeProperty>',
            '    ',
            '    <owl:DatatypeProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/order">',
            '        <rdfs:label xml:lang="en">order</rdfs:label>',
            '        <rdfs:comment xml:lang="en">The presentation order of an element</rdfs:comment>',
            '        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#decimal"/>',
            '    </owl:DatatypeProperty>',
            '    ',
            '    <owl:DatatypeProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/isClosed">',
            '        <rdfs:label xml:lang="en">is closed</rdfs:label>',
            '        <rdfs:comment xml:lang="en">Indicates whether a dimension has a closed set of members</rdfs:comment>',
            '        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#boolean"/>',
            '    </owl:DatatypeProperty>',
            '    ',
            '    <owl:DatatypeProperty rdf:about="https://xbrl.ifrs.org/ontology/ifrs-sds/arcrole">',
            '        <rdfs:label xml:lang="en">arcrole</rdfs:label>',
            '        <rdfs:comment xml:lang="en">The XBRL arcrole URI that defines the relationship type</rdfs:comment>',
            '        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#anyURI"/>',
            '    </owl:DatatypeProperty>',
            '    '
        ]
    
    def generate_role_instances(self):
        """Generate role instances"""
        lines = ['    <!-- Role Instances -->', '    ']
        
        for role_id, role_data in self.roles.items():
            lines.extend([
                f'    <ifrs-sds:Role rdf:about="{role_data["uri"]}">',
                f'        <rdf:type rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/Role"/>',
                f'        <ifrs-sds:roleURI rdf:datatype="http://www.w3.org/2001/XMLSchema#anyURI">{self.escape_xml(role_data["uri"])}</ifrs-sds:roleURI>',
            ])
            
            if role_data.get('definition'):
                lines.append(f'        <ifrs-sds:roleDefinition xml:lang="en">{self.escape_xml(role_data["definition"])}</ifrs-sds:roleDefinition>')
            
            # Add element references from generic reference linkbase
            role_uri = role_data["uri"]
            if role_uri in self.role_references:
                for ref in self.role_references[role_uri]:
                    # Extract concept from the reference
                    # Note: This is simplified - in reality might need more parsing
                    lines.append(f'        <!-- Generic element reference: {ref["arcrole"]} -->')
            
            lines.extend(['    </ifrs-sds:Role>', '    '])
        
        return lines
    
    def generate_concept_instances(self):
        """Generate concept instances"""
        lines = ['    <!-- Concept Instances -->', '    ']
        
        # Limit to first 100 concepts for manageable output
        concept_count = 0
        max_concepts = 500
        
        for concept_name, concept_data in self.concepts.items():
            if concept_count >= max_concepts:
                lines.append(f'    <!-- ... {len(self.concepts) - max_concepts} more concepts ... -->')
                break
            
            concept_class = self.determine_concept_class(concept_data)
            uri = concept_data['uri']
            
            lines.extend([
                f'    <ifrs-sds:{concept_class} rdf:about="{uri}">',
                f'        <rdf:type rdf:resource="https://xbrl.ifrs.org/ontology/ifrs-sds/{concept_class}"/>',
                f'        <ifrs-sds:conceptName>{self.escape_xml(concept_name)}</ifrs-sds:conceptName>',
            ])
            
            # Add label if available
            if concept_name in self.labels:
                for lang, label_text in self.labels[concept_name].items():
                    lines.append(f'        <rdfs:label xml:lang="{lang}">{self.escape_xml(label_text)}</rdfs:label>')
            
            # Add documentation if available
            if concept_name in self.documentation:
                for lang, doc_text in self.documentation[concept_name].items():
                    lines.append(f'        <ifrs-sds:documentation xml:lang="{lang}">{self.escape_xml(doc_text)}</ifrs-sds:documentation>')
            
            # Add inline labels if available
            if concept_name in self.inline_labels:
                for role_type, lang_dict in self.inline_labels[concept_name].items():
                    for lang, label_text in lang_dict.items():
                        lines.append(f'        <ifrs-sds:inlineLabel xml:lang="{lang}">{self.escape_xml(label_text)}</ifrs-sds:inlineLabel>')
            
            # Add parent-child relationships
            for arc in self.presentation_arcs:
                if arc['child'] == concept_name:
                    parent_uri = f"https://xbrl.ifrs.org/ontology/ifrs-sds/{arc['parent']}"
                    lines.append(f'        <ifrs-sds:hasParent rdf:resource="{parent_uri}"/>')
                    lines.append(f'        <ifrs-sds:order rdf:datatype="http://www.w3.org/2001/XMLSchema#decimal">{arc["order"]}</ifrs-sds:order>')
            
            # Add dimensional relationships
            for arc in self.definition_arcs:
                arcrole = arc['arcrole']
                
                # Dimension-domain relationship
                if 'dimension-domain' in arcrole and arc['from'] == concept_name:
                    target_uri = f"https://xbrl.ifrs.org/ontology/ifrs-sds/{arc['to']}"
                    lines.append(f'        <ifrs-sds:hasDomain rdf:resource="{target_uri}"/>')
                
                # Domain-member relationship
                elif 'domain-member' in arcrole and arc['from'] == concept_name:
                    target_uri = f"https://xbrl.ifrs.org/ontology/ifrs-sds/{arc['to']}"
                    lines.append(f'        <ifrs-sds:hasMember rdf:resource="{target_uri}"/>')
                
                # Hypercube-dimension relationship
                elif 'hypercube-dimension' in arcrole and arc['from'] == concept_name:
                    target_uri = f"https://xbrl.ifrs.org/ontology/ifrs-sds/{arc['to']}"
                    lines.append(f'        <ifrs-sds:hasDimension rdf:resource="{target_uri}"/>')
                
                # All relationship (primary item to hypercube)
                elif '/all' in arcrole and arc['to'] == concept_name:
                    source_uri = f"https://xbrl.ifrs.org/ontology/ifrs-sds/{arc['from']}"
                    lines.append(f'        <ifrs-sds:hasHypercube rdf:resource="{source_uri}"/>')
                    if arc.get('closed'):
                        lines.append(f'        <ifrs-sds:isClosed rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</ifrs-sds:isClosed>')
            
            # Add dimensional relationships from dimensions/ folder
            for arc in self.dimension_arcs:
                arcrole = arc['arcrole']
                
                # Dimension-domain relationship
                if 'dimension-domain' in arcrole and arc['from'] == concept_name:
                    target_uri = f"https://xbrl.ifrs.org/ontology/ifrs-sds/{arc['to']}"
                    lines.append(f'        <ifrs-sds:hasDomain rdf:resource="{target_uri}"/>')
                
                # Domain-member relationship
                elif 'domain-member' in arcrole and arc['from'] == concept_name:
                    target_uri = f"https://xbrl.ifrs.org/ontology/ifrs-sds/{arc['to']}"
                    lines.append(f'        <ifrs-sds:hasMember rdf:resource="{target_uri}"/>')
                
                # Hypercube-dimension relationship
                elif 'hypercube-dimension' in arcrole and arc['from'] == concept_name:
                    target_uri = f"https://xbrl.ifrs.org/ontology/ifrs-sds/{arc['to']}"
                    lines.append(f'        <ifrs-sds:hasDimension rdf:resource="{target_uri}"/>')
                
                # All relationship (primary item to hypercube)
                elif '/all' in arcrole and arc['from'] == concept_name:
                    target_uri = f"https://xbrl.ifrs.org/ontology/ifrs-sds/{arc['to']}"
                    lines.append(f'        <ifrs-sds:hasHypercube rdf:resource="{target_uri}"/>')
                    if arc.get('closed'):
                        lines.append(f'        <ifrs-sds:isClosed rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</ifrs-sds:isClosed>')
            
            # Mark if concept has references
            if concept_name in self.references:
                lines.append(f'        <ifrs-sds:hasReference rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</ifrs-sds:hasReference>')
            
            lines.extend(['    </ifrs-sds:{}>'.format(concept_class), '    '])
            concept_count += 1
        
        return lines
    
    def escape_xml(self, text):
        """Escape XML special characters"""
        if not text:
            return ''
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&apos;'))
    
    def process_taxonomy(self):
        """Process entire taxonomy and generate RDF"""
        print(f"Processing XBRL taxonomy from: {self.taxonomy_path}")
        
        # Find and parse core schema
        core_schema = list(self.taxonomy_path.glob('*-cor_*.xsd'))
        if core_schema:
            self.parse_schema_file(core_schema[0])
        
        # Process linkbases
        linkbase_path = self.taxonomy_path / 'linkbases'
        
        if linkbase_path.exists():
            # Process each standard (s1, s2, etc.)
            for std_dir in linkbase_path.iterdir():
                if std_dir.is_dir():
                    print(f"\nProcessing standard: {std_dir.name}")
                    
                    # Parse role definitions
                    for role_file in std_dir.glob('rol_*.xsd'):
                        self.parse_role_file(role_file)
                    
                    # Parse presentation linkbases
                    for pres_file in std_dir.glob('pre_*.xml'):
                        self.d(pres_file)
                    
                    # Parse definition linkbases
                    for def_file in std_dir.glob('def_*.xml'):
                        self.parse_definition_linkbase(def_file)
                    
                    # Parse generic label linkbases
                    for gla_file in std_dir.glob('gla_*.xml'):
                        self.parse_label_linkbase(gla_file)
                    
                    # Parse reference linkbases
                    for ref_file in std_dir.glob('ref_*.xml'):
                        self.parse_reference_linkbase(ref_file)
                    
                    # Parse generic reference linkbases (gre files)
                    for gre_file in std_dir.glob('gre_*.xml'):
                        self.parse_generic_reference_linkbase(gre_file)
        
        # Process label files
        labels_path = self.taxonomy_path / 'labels'
        if labels_path.exists():
            for label_file in labels_path.glob('lab_*.xml'):
                self.parse_label_linkbase(label_file)
            
            for doc_file in labels_path.glob('doc_*.xml'):
                self.parse_label_linkbase(doc_file)
            
            # Parse inline label files (in_ files)
            for in_file in labels_path.glob('in_*.xml'):
                self.parse_inline_label_linkbase(in_file)
        
        # Process dimensions folder
        dimensions_path = self.taxonomy_path / 'dimensions'
        if dimensions_path.exists():
            print(f"\nProcessing dimensions folder: {dimensions_path}")
            
            # Parse role definitions in dimensions folder
            for role_file in dimensions_path.glob('rol_*.xsd'):
                self.parse_role_file(role_file)
            
            # Parse presentation linkbases in dimensions folder
            for pres_file in dimensions_path.glob('pre_*.xml'):
                self.parse_presentation_linkbase(pres_file)
            
            # Parse dimension linkbases (dim files)
            for dim_file in dimensions_path.glob('dim_*.xml'):
                self.parse_dimension_linkbase(dim_file)
            
            # Parse generic reference linkbases in dimensions folder
            for gre_file in dimensions_path.glob('gre_*.xml'):
                self.parse_generic_reference_linkbase(gre_file)
        
        # Generate RDF
        rdf_content = self.generate_rdf_ontology()
        
        # Write to file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(rdf_content)
        
        print(f"\n✓ RDF ontology generated: {self.output_file}")
        print(f"  - Concepts: {len(self.concepts)}")
        print(f"  - Roles: {len(self.roles)}")
        print(f"  - Presentation arcs: {len(self.presentation_arcs)}")
        print(f"  - Definition arcs: {len(self.definition_arcs)}")
        print(f"  - Dimension arcs: {len(self.dimension_arcs)}")
        print(f"  - Labels: {len(self.labels)}")
        print(f"  - Inline labels: {len(self.inline_labels)}")
        print(f"  - Documentation: {len(self.documentation)}")
        print(f"  - Role references: {len(self.role_references)}")


def main(taxonomy_path, output):
    # Check if path exists
    if not os.path.exists(taxonomy_path):
        print(f"Error: Taxonomy path not found: {taxonomy_path}")
        return 1
    
    # Create converter and process
    converter = XBRLToRDFConverter(taxonomy_path, output)
    converter.process_taxonomy()
    
    return 0


if __name__ == "__main__":
    MAIN_DIR = "./data/ifrs_sds/ifrs_sds"
    STANDARDS = ['ifrs_s1', 'ifrs_s2']
    OUTPUT_FILE = "./data/ifrs_sds_ontology.rdf"
    LABEL_DIR = os.path.join(MAIN_DIR, "labels")

    # print("Initializing graph ...")
    # g = Graph()
    # g.bind("ifrs", IFRS)
    # g.bind("skos", SKOS)
    # g.bind("dcterms", DCTERMS)
    # g.bind("rdfs", RDFS)

    # for standard in STANDARDS:
    #     taxonomy_dir = os.path.join(MAIN_DIR, f"linkbases/{standard}")
    #     parse_taxonomy_folder(g, taxonomy_dir)

    # parse_labels(g, LABEL_DIR)
    # # Serialize RDF Graph
    # g.serialize(destination=OUTPUT_FILE, format="turtle")
    # print(f"RDF/Turtle file generated: {OUTPUT_FILE}")


    main(MAIN_DIR, OUTPUT_FILE)
