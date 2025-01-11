import streamlit as st
import pandas as pd
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import re  
import base64
import numpy as np

indexName = 'compounds'

# Use environment variables for sensitive data
es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "*nLVGSfCz+3jr1OsIVeA"),
    ca_certs="/Users/judepops/Documents/PathIntegrate/Code/Processing/semantic_search/elasticsearch-8.13.2/config/certs/http_ca.crt"
)

def search(input_keyword):
    model = SentenceTransformer('all-mpnet-base-v2')
    vector_of_input_keyword = model.encode(input_keyword)
    query = {
        "field": "NAME_VECTOR",
        "query_vector": vector_of_input_keyword,
        "k": 2,
        "num_candidates": 500,
    }
    res = es.knn_search(index=indexName, knn=query, source=['COMPOUND_ID', 'NAME', 'TYPE', 'SOURCE'])
    return res["hits"]["hits"]

def add_bg_from_local():
    bg_image = get_base64_of_file('graphene.png')
    st.markdown(f"""
    <style>
    /* Full Page Background */
    body {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        margin: 0 !important;
        padding: 0 !important;
    }}
    /* Streamlit Container Adjustment */
    .stApp {{
        background-color: transparent;
    }}
    </style>
    """, unsafe_allow_html=True)

def img_to_base64(img_path):
    import base64
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def get_base64_of_file(file_path):
    """Read binary file and convert to base64 encoded string."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def main():

    sidebar_bg = get_base64_of_file('sidebar_bg.jpg')

    # Add background image
    add_bg_from_local()

    # Add background image to the sidebar and custom styles via markdown
    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        background-image: url("data:image/jpg;base64,{sidebar_bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    
    .main-title {{
        font-size:50px;
        font-weight:bold;
        color: #2e6e9e;  /* Main title color */
    }}
    .yellow-title {{
        font-size:35px;
        font-weight:bold;
        color: #dad234;  /* CSS color for yellow */
    }}
    .citation {{
        font-style: italic;  /* Italicize the citation */
        margin-top: 5px;  /* Space between the image and the citation */
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-title">Find Your Metabolite ChEBI ID!</p>', unsafe_allow_html=True)

    # Load and display the logo image below the main title
    logo_path = 'Chebi_logo_image.png'
    st.image(logo_path, caption=None, width=100) 

    # Display the citation in italics right below the image
    st.markdown('<div class="citation">Hastings J, Owen G, Dekker A, Ennis M, Kale N, Muthukrishnan V, Turner S, Swainston N, Mendes P, Steinbeck C. (2016). ChEBI in 2016: Improved services and an expanding collection of metabolites. Nucleic Acids Res.</div>', unsafe_allow_html=True)
    st.markdown('')
    
    # Sidebar navigation dropdown
    st.sidebar.markdown('<p class="yellow-title">Select Tool</p>', unsafe_allow_html=True)
    page_options = ["About", "Single Search", "Multi Search"]
    if 'active_page' not in st.session_state:
        st.session_state.active_page = 'About'

    # Update active page based on selection
    st.session_state.active_page = st.sidebar.selectbox("Choose a page:", page_options)

    # Display the appropriate page based on the current state
    if st.session_state.active_page == 'About':
        st.markdown('<p class="yellow-title">About</p>', unsafe_allow_html=True)
        st.write("This is an online tool to infer the ChEBI ID of a chemical compound using its IUPAC name."
                 " The tool will calculate the vector embedding of the IUPAC name and use a pre-trained BERT LLM model to identify"
                 " the corresponding ChEBI ID through a semantic search powered by a KNN search and l2 normalization.")
        st.markdown('')
        st.image('semantic_worflow.jpg', caption=None) 
        st.markdown('')
        st.markdown('**Figure 1: Semantic Search Workflow (left to right)**. This image displays the general workflow algorithm behind this website. It begins with the transformation of ChEBI database metabolites into a BERT vector representation, followed by their indexing in an online database powered by elastic search. This database can be efficiently queried with a metabolite name using semantic searching powered by l2 normalisation of the BERT LLM vectors.')


    elif st.session_state.active_page == 'Single Search':
        st.markdown('<p class="yellow-title">Single Search</p>', unsafe_allow_html=True)
        with st.form(key='search_form'):
            search_query = st.text_input("Enter your search query")
            submit_button = st.form_submit_button("Search")
        if submit_button and search_query:
            results = search(search_query)
            if results:
                st.subheader("Search Results")
                display_search_results(results)


    elif st.session_state.active_page == 'Multi Search':
        st.markdown('<p class="yellow-title">Multi Search</p>', unsafe_allow_html=True)

        # Allow the user to set the threshold before searching
        threshold = st.slider("Set Match Similarity Threshold (in percentage)", min_value=0, max_value=100, value=75)
        threshold /= 100  # Convert percentage to decimal for comparison

        all_results = []  # Initialize all_results at the top

        with st.form(key='list_form'):
            # Updated instruction to reflect that each ID must be on a new line
            id_list = st.text_area("Enter a list of Compound IDs (each ID on a new line)")
            submit_button = st.form_submit_button("Search")

        if submit_button and id_list:
            # Using newlines to split the compound IDs
            query_ids = id_list.strip().split('\n')

            for compound_id in query_ids:
                input_id = compound_id.strip()  # Store the original input ID
                if input_id:  # Check if the input_id is not empty
                    results = search(input_id)
                    if results:
                        # Access the hits from the result JSON
                        hits = results

                        # Initialize variables to store the best hits
                        best_hit = None
                        best_score = -np.inf
                        best_is_kegg = False  # Track if the best hit is a KEGG COMPOUND

                        # Check each hit to determine the best match
                        for hit in hits:
                            score = hit['_score']  # Assuming the relevance score is in '_score'
                            if score <= threshold:
                                continue  # Skip if the score is below the threshold

                            compound_id = str(hit['_source']['COMPOUND_ID']).strip()  # Ensure compound_id is a stripped string
                            source = hit['_source'].get('SOURCE', 'N/A')
                            is_kegg = source == 'KEGG COMPOUND'

                            # Determine if the current hit is better
                            if (best_hit is None) or (score > best_score) or \
                            (score == best_score and is_kegg and not best_is_kegg):
                                best_hit = hit
                                best_score = score
                                best_is_kegg = is_kegg

                        # Assuming you want the top match for each compound
                        if best_hit:  # Check if there is a valid best match
                            matched_name = best_hit['_source']['NAME']
                            matched_compound_id = best_hit['_source']['COMPOUND_ID']
                            compound_source = best_hit['_source'].get('SOURCE', 'N/A')
                            comp_type = best_hit['_source'].get('TYPE', 'N/A')

                            # Append the result to the list
                            all_results.append({
                                "Input": input_id,  # Use the original input ID here
                                "COMPOUND_ID": matched_compound_id,
                                "NAME": matched_name,
                                "TYPE": comp_type,
                                "SOURCE": compound_source,
                                "Match Similarity Score": f"{best_score * 100:.0f}%"  # Format score as a percentage
                            })
                        else:
                            all_results.append({
                                "Input": input_id,  # Use the original input ID here
                                "COMPOUND_ID": "No match above threshold",
                                "NAME": "No match",
                                "TYPE": "No match",
                                "SOURCE": "N/A",
                                "Match Similarity Score": "0%"
                            })

        if len(all_results) > 0:  # Safely check if all_results has data
            df = pd.DataFrame(all_results)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "multi_search_results.csv", "text/csv")
            st.subheader("Preview of Search Results")
            df['COMPOUND_ID'] = df['COMPOUND_ID'].apply(lambda x: str(x).replace(',', ''))
            st.write(df)
        else:
            st.write("No results found for any of the input IDs.")


def display_search_results(results):
    for result in results:
        with st.expander(f"Compound Match: {result['_source'].get('NAME', 'N/A')}"):
            chebi_id = result['_source'].get('COMPOUND_ID', 'N/A')
            id_type = result['_source'].get('TYPE', 'N/A')
            match_score = result['_score']
            image_url = f"https://www.ebi.ac.uk/chebi/displayImage.do?defaultImage=true&chebiId={chebi_id}"
            
            # Display ChEBI image
            st.image(image_url, caption=f"ChEBI Image for {chebi_id}")
            
            # Display ChEBI ID and link to ChEBI page
            st.markdown(f"**ChEBI ID**: {chebi_id}")
            st.markdown(f"**Match Similarity Score**: {'{:.0f}'.format(match_score * 100)}%")
            st.markdown(f"**Synonym or IUPAC Name?**: {id_type}")

            ebi_url = f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:{chebi_id}"
            st.markdown(f"[View on EBI]({ebi_url})")

if __name__ == "__main__":
    main()