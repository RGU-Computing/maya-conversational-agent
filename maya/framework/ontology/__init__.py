"""Runtime access to the main partner ontology and its memory contracts."""

from maya.framework.ontology.loader import load_ontology
from maya.framework.ontology.registry import OntologyRegistry, get_ontology_registry

__all__ = [
    "OntologyRegistry",
    "get_ontology_registry",
    "load_ontology",
]
