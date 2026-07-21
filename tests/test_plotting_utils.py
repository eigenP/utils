import os
import xml.etree.ElementTree as ET
import pytest
import matplotlib.pyplot as plt
from pathlib import Path

from eigenp_utils.plotting_utils import savefig_svg

def test_savefig_svg(tmp_path):
    # Set up a test figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title('Test Plot')
    fig.suptitle('My Awesome Title')

    # Define file path
    svg_path = tmp_path / "test_figure"

    # Call the function
    savefig_svg(svg_path, bgnd_color=(1, 0, 0, 0.5), pad_inches=0.2)

    # Make sure we add '.svg' if it's missing in test assertion
    svg_file = str(svg_path) + ".svg"

    # Assert file exists
    assert os.path.exists(svg_file)

    # Read the SVG content
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # SVG namespaces are generally used
    namespaces = {'dc': 'http://purl.org/dc/elements/1.1/',
                  'cc': 'http://creativecommons.org/ns#',
                  'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                  'svg': 'http://www.w3.org/2000/svg'}

    # Check for metadata
    # The exact structure depends on matplotlib's SVG backend, but Dublin Core properties are typically nested inside <metadata><rdf:RDF><cc:Work>
    metadata_elem = root.find('svg:metadata', namespaces)
    assert metadata_elem is not None, "Metadata element not found in SVG"

    rdf_work = metadata_elem.find('.//cc:Work', namespaces)
    assert rdf_work is not None, "cc:Work element not found in metadata"

    dc_title = rdf_work.find('.//dc:title', namespaces)
    assert dc_title is not None, "dc:title not found"
    assert dc_title.text == 'My Awesome Title', f"Expected title 'My Awesome Title', got '{dc_title.text}'"

    dc_date = rdf_work.find('.//dc:date', namespaces)
    assert dc_date is not None, "dc:date not found"
    # Date should be an ISO format string
    assert len(dc_date.text) > 0, "Date string is empty"

    dc_creator = rdf_work.find('.//dc:creator//cc:Agent//dc:title', namespaces)
    assert dc_creator is not None, "dc:creator not found"
    assert dc_creator.text == 'eigenp', f"Expected creator 'eigenp', got '{dc_creator.text}'"

    plt.close('all')

def test_savefig_svg_no_suptitle(tmp_path):
    # Set up a test figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    # Define file path
    svg_path = tmp_path / "test_figure_no_title"

    # Call the function
    savefig_svg(svg_path)

    # Make sure we add '.svg' if it's missing in test assertion
    svg_file = str(svg_path) + ".svg"

    # Assert file exists
    assert os.path.exists(svg_file)

    # Read the SVG content
    tree = ET.parse(svg_file)
    root = tree.getroot()

    namespaces = {'dc': 'http://purl.org/dc/elements/1.1/',
                  'cc': 'http://creativecommons.org/ns#',
                  'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                  'svg': 'http://www.w3.org/2000/svg'}

    metadata_elem = root.find('svg:metadata', namespaces)
    rdf_work = metadata_elem.find('.//cc:Work', namespaces)
    dc_title = rdf_work.find('.//dc:title', namespaces)

    # If no suptitle exists, it should default to the filename string
    assert dc_title is not None
    assert dc_title.text == str(svg_path)

    plt.close('all')

def test_set_plotting_style():
    from eigenp_utils.plotting_utils import set_plotting_style

    # Backup original rcParams
    orig_fonttype = plt.rcParams['svg.fonttype']
    orig_unicode = plt.rcParams['axes.unicode_minus']

    try:
        # Run the function with default editable_text=True
        set_plotting_style()

        # Check that settings are applied
        assert plt.rcParams['font.family'] == ['sans-serif']
        assert plt.rcParams['svg.fonttype'] == 'none'
        assert plt.rcParams['axes.unicode_minus'] == False

        # Reset and test with editable_text=False
        # Our updated set_plotting_style explicitly sets these back to default
        set_plotting_style(editable_text=False)
        assert plt.rcParams['svg.fonttype'] == 'path'
        assert plt.rcParams['axes.unicode_minus'] == True

    finally:
        # Restore globally to avoid polluting other tests
        plt.rcParams['svg.fonttype'] = orig_fonttype
        plt.rcParams['axes.unicode_minus'] = orig_unicode

def test_savefig_svg_editable_text(tmp_path):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "Test Editable Text")

    # Test default editable_text=True
    svg_path_true = tmp_path / "editable_true"
    savefig_svg(svg_path_true) # Should use default True

    # Check that settings weren't permanently changed
    assert plt.rcParams['svg.fonttype'] != 'none'

    # Read the SVG
    svg_file_true = str(svg_path_true) + ".svg"
    with open(svg_file_true, 'r') as f:
        content_true = f.read()

    # Text elements should be standard <text> in SVG if fonttype='none'
    assert "<text" in content_true, "Expected <text> elements when editable_text=True"

    # Test editable_text=False
    svg_path_false = tmp_path / "editable_false"
    savefig_svg(svg_path_false, editable_text=False)

    svg_file_false = str(svg_path_false) + ".svg"
    with open(svg_file_false, 'r') as f:
        content_false = f.read()

    # If fonttype='path' (the default), text is converted to paths, so <text> elements usually won't exist
    assert "<text" not in content_false, "Did not expect <text> elements when editable_text=False"

    plt.close('all')
