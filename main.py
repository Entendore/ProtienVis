import streamlit as st
import plotly.express as px
import plotly.io as pio
import pandas as pd
import re
import io
import base64
from Bio import Align
from Bio.SeqUtils import MeltingTemp as mt

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced DNA Analysis Suite",
    page_icon="🧬",
    layout="wide"
)

# --- Data Dictionaries (Defined at top for efficiency) ---
CODON_TABLE = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L', 'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': 'Stop', 'UAG': 'Stop', 'UGU': 'C', 'UGC': 'C', 'UGA': 'Stop', 'UGG': 'W',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L', 'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M', 'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V', 'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}
AMINO_ACID_DETAILS = {
    'A': {'name': 'Alanine', 'type': 'Non-polar, Hydrophobic'}, 'R': {'name': 'Arginine', 'type': 'Positively Charged'},
    'N': {'name': 'Asparagine', 'type': 'Polar, Uncharged'}, 'D': {'name': 'Aspartic Acid', 'type': 'Negatively Charged'},
    'C': {'name': 'Cysteine', 'type': 'Polar, Uncharged'}, 'Q': {'name': 'Glutamine', 'type': 'Polar, Uncharged'},
    'E': {'name': 'Glutamic Acid', 'type': 'Negatively Charged'}, 'G': {'name': 'Glycine', 'type': 'Non-polar, Hydrophobic'},
    'H': {'name': 'Histidine', 'type': 'Positively Charged'}, 'I': {'name': 'Isoleucine', 'type': 'Non-polar, Hydrophobic'},
    'L': {'name': 'Leucine', 'type': 'Non-polar, Hydrophobic'}, 'K': {'name': 'Lysine', 'type': 'Positively Charged'},
    'M': {'name': 'Methionine', 'type': 'Non-polar, Hydrophobic'}, 'F': {'name': 'Phenylalanine', 'type': 'Non-polar, Hydrophobic'},
    'P': {'name': 'Proline', 'type': 'Non-polar, Hydrophobic'}, 'S': {'name': 'Serine', 'type': 'Polar, Uncharged'},
    'T': {'name': 'Threonine', 'type': 'Polar, Uncharged'}, 'W': {'name': 'Tryptophan', 'type': 'Non-polar, Hydrophobic'},
    'Y': {'name': 'Tyrosine', 'type': 'Polar, Uncharged'}, 'V': {'name': 'Valine', 'type': 'Non-polar, Hydrophobic'},
    'Stop': {'name': 'Stop Codon', 'type': 'Translation Termination'}
}
RESTRICTION_ENZYMES = {
    "EcoRI": "GAATTC", "HindIII": "AAGCTT", "BamHI": "GGATCC", "XbaI": "TCTAGA", "NotI": "GCGGCCGC",
    "KpnI": "GGTACC", "SacI": "GAGCTC", "SmaI": "CCCGGG", "PstI": "CTGCAG", "SalI": "GTCGAC",
    "EcoRV": "GATATC", "SphI": "GCATGC", "NcoI": "CCATGG", "BglII": "AGATCT", "HpaI": "GTTAAC",
    "PvuII": "CAGCTG", "ClaI": "ATCGAT", "MluI": "ACGCGT", "NdeI": "CATATG", "NheI": "GCTAGC",
    "Bsu36I": "CCGCTC", "BsuEI": "CCTNAGG"
}
KYTE_DOOLITTLE_SCALE = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7,
    'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5,
    'N': -3.5, 'K': -3.9, 'R': -4.5
}

# --- Core Biological Logic (with Caching) ---

@st.cache_data
def validate_dna_sequence(sequence: str) -> bool:
    """Validates if a sequence contains only A, T, C, G."""
    sequence = sequence.upper()
    if not sequence:
        return False
    allowed_bases = {'A', 'T', 'C', 'G', 'N'}
    if not all(base in allowed_bases for base in sequence):
        return False
    return True

@st.cache_data
def transcribe(_dna_sequence: str) -> str:
    """Transcribes a DNA sequence into an mRNA sequence."""
    transcription_map = {'A': 'U', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return "".join([transcription_map.get(base, 'X') for base in _dna_sequence.upper()])

@st.cache_data
def translate(_mrna_sequence: str, reading_frame: int) -> tuple[list, str]:
    """Translates an mRNA sequence into a protein sequence."""
    protein_details = []
    protein_sequence = []
    mrna_to_translate = _mrna_sequence[reading_frame-1:]
    for i in range(0, len(mrna_to_translate), 3):
        codon = mrna_to_translate[i:i+3]
        if len(codon) == 3:
            amino_acid_letter = CODON_TABLE.get(codon, 'Unknown')
            details = AMINO_ACID_DETAILS.get(amino_acid_letter, {'name': 'Unknown', 'type': 'N/A'})
            protein_details.append({
                "mRNA Codon": codon, "Amino Acid (Letter)": amino_acid_letter,
                "Full Name": details['name'], "Properties": details['type']
            })
            if amino_acid_letter == 'Stop': break
            protein_sequence.append(amino_acid_letter)
    return protein_details, "".join(protein_sequence)

@st.cache_data
def calculate_sequence_stats(_dna_sequence: str) -> dict:
    """Calculates basic statistics for a DNA sequence."""
    stats = {'Length': len(_dna_sequence), 'A Count': _dna_sequence.upper().count('A'), 'T Count': _dna_sequence.upper().count('T'), 'C Count': _dna_sequence.upper().count('C'), 'G Count': _dna_sequence.upper().count('G')}
    gc_count = stats['G Count'] + stats['C Count']
    stats['GC Content (%)'] = round((gc_count / stats['Length']) * 100, 2) if stats['Length'] > 0 else 0
    return stats

@st.cache_data
def reverse_complement(_dna_sequence: str) -> str:
    """Returns the reverse complement of a DNA sequence."""
    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return "".join([complement_map.get(base, 'X') for base in _dna_sequence.upper()[::-1]])

@st.cache_data
def find_orfs(_dna_sequence: str, min_len: int = 100) -> list:
    """Finds all Open Reading Frames (ORFs) in all 6 reading frames."""
    start_codon = "ATG"
    stop_codons = ["TAA", "TAG", "TGA"]
    orfs = []
    strands = {'+': _dna_sequence.upper(), '-': reverse_complement(_dna_sequence)}
    for strand, seq in strands.items():
        for frame in range(3):
            for i in range(frame, len(seq) - 2, 3):
                codon = seq[i:i+3]
                if codon == start_codon:
                    for j in range(i + 3, len(seq) - 2, 3):
                        stop_codon = seq[j:j+3]
                        if stop_codon in stop_codons:
                            orf_seq = seq[i:j+3]
                            if len(orf_seq) >= min_len:
                                protein = translate(transcribe(orf_seq), 1)[1]
                                orfs.append({
                                    "Strand": strand, "Frame": f"+{frame+1}" if strand == '+' else f"-{frame+1}",
                                    "Start": i + 1, "End": j + 3, "Length": len(orf_seq), "Protein": protein
                                })
                            break
    return orfs

def parse_fasta(file) -> dict:
    """Parses a FASTA file and returns a dictionary of {header: sequence}."""
    fasta_dict = {}
    try:
        content = file.read().decode("utf-8")
        records = content.split('>')[1:]
        for record in records:
            parts = record.split('\n', 1)
            header = parts[0].strip()
            sequence = "".join(parts[1].split()).upper()
            if header and sequence:
                fasta_dict[header] = sequence
    except Exception as e:
        st.error(f"Error parsing FASTA file: {e}")
        return None
    return fasta_dict

@st.cache_data
def design_primers(_sequence: str, target_start: int, target_end: int, primer_length: int = 20) -> dict:
    """Designs forward and reverse primers for a given target region."""
    target_start = max(0, target_start)
    target_end = min(len(_sequence), target_end)
    if target_start >= target_end:
        return {"error": "Invalid target region. Start must be less than End."}
    fp_start = max(0, target_start - primer_length)
    fp_end = target_start
    forward_primer = _sequence[fp_start:fp_end]
    rc_sequence = reverse_complement(_sequence)
    rp_start_on_rc = len(_sequence) - target_end
    rp_end_on_rc = rp_start_on_rc + primer_length
    reverse_primer = rc_sequence[rp_start_on_rc:rp_end_on_rc]
    def calculate_primer_properties(primer):
        if not primer: return None
        tm = mt.Tm_Wallace(primer)
        gc = (primer.count('G') + primer.count('C')) / len(primer) * 100
        return {"sequence": primer, "length": len(primer), "tm": round(tm, 2), "gc": round(gc, 2)}
    return {
        "forward_primer": calculate_primer_properties(forward_primer),
        "reverse_primer": calculate_primer_properties(reverse_primer),
        "target_region": f"{target_start+1}-{target_end}"
    }

@st.cache_data
def find_restriction_sites(_sequence: str, enzymes: dict) -> list:
    """Finds all restriction enzyme sites in a DNA sequence."""
    found_sites = []
    for enzyme, site in enzymes.items():
        regex_site = site.replace('N', '.')
        for match in re.finditer(regex_site, _sequence):
            start_pos = match.start() + 1 # 1-based indexing
            found_sites.append({
                "Enzyme": enzyme,
                "Recognition Site": site,
                "Cut Position": f"{start_pos}-{start_pos + len(site) - 1}",
                "Sequence Context": f"...{_sequence[max(0, start_pos-10):start_pos-1]}**{_sequence[start_pos-1:start_pos+len(site)-1]}**{_sequence[start_pos+len(site)-1:start_pos+len(site)+9]}..."
            })
    return sorted(found_sites, key=lambda x: int(x['Cut Position'].split('-')[0]))

@st.cache_data
def calculate_sliding_gc(_sequence: str, window_size: int, step_size: int) -> pd.DataFrame:
    """Calculates GC content in a sliding window across a sequence."""
    results = []
    for i in range(0, len(_sequence) - window_size + 1, step_size):
        window = _sequence[i:i + window_size]
        gc_count = window.count('G') + window.count('C')
        gc_percent = (gc_count / window_size) * 100 if window_size > 0 else 0
        position = i + window_size // 2
        results.append({"Position": position, "GC Content (%)": round(gc_percent, 2)})
    return pd.DataFrame(results)

@st.cache_data
def calculate_hydropathy(_protein_sequence: str, window_size: int) -> pd.DataFrame:
    """Calculates the average Kyte-Doolittle hydropathy score in a sliding window."""
    results = []
    seq_len = len(_protein_sequence)
    if seq_len < window_size:
        return pd.DataFrame()
    for i in range(seq_len - window_size + 1):
        window = _protein_sequence[i:i + window_size]
        score = sum(KYTE_DOOLITTLE_SCALE.get(aa, 0) for aa in window) / window_size
        position = i + window_size // 2
        results.append({"Position": position, "Hydropathy Score": round(score, 3)})
    return pd.DataFrame(results)

# --- FIX: Use Plotly's built-in image export ---
def fig_to_base64(fig):
    """Converts a Plotly figure to a base64 encoded string using Kaleido."""
    # Ensure kaleido is installed for image export
    try:
        image_bytes = pio.to_image(fig, format="png", engine="kaleido")
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        st.error(f"Could not generate image for report. Please ensure 'kaleido' is installed (`pip install kaleido`). Error: {e}")
        return ""


# --- Streamlit App UI ---

st.title("🧬 Advanced DNA Analysis Suite")

# --- Session State Initialization ---
if 'sequence_data' not in st.session_state:
    st.session_state.sequence_data = {}
if 'current_sequence' not in st.session_state:
    st.session_state.current_sequence = ""
if 'current_header' not in st.session_state:
    st.session_state.current_header = ""

# --- Sidebar for Input and Statistics ---
st.sidebar.header("📂 Input & Statistics")
uploaded_file = st.sidebar.file_uploader("Upload a FASTA file", type=["fasta", "fa", "txt"])

if uploaded_file is not None:
    parsed_data = parse_fasta(uploaded_file)
    if parsed_data:
        st.session_state.sequence_data = parsed_data
        st.session_state.current_header = list(parsed_data.keys())[0]
        st.session_state.current_sequence = parsed_data[st.session_state.current_header]
        st.sidebar.success(f"Successfully loaded {len(parsed_data)} sequence(s).")

if st.session_state.sequence_data:
    selected_header = st.sidebar.selectbox(
        "Select a sequence to analyze:",
        options=list(st.session_state.sequence_data.keys()),
        index=list(st.session_state.sequence_data.keys()).index(st.session_state.current_header),
        key="sequence_selector"
    )
    st.session_state.current_header = selected_header
    st.session_state.current_sequence = st.session_state.sequence_data[selected_header]

if st.session_state.current_sequence:
    stats = calculate_sequence_stats(st.session_state.current_sequence)
    st.sidebar.subheader("📊 Sequence Statistics")
    st.sidebar.metric("Length", stats['Length'])
    st.sidebar.metric("GC Content", f"{stats['GC Content (%)']}%")
    with st.sidebar.expander("Nucleotide Counts"):
        st.json({k: v for k, v in stats.items() if 'Count' in k})

# --- Main Content Area ---
if not st.session_state.current_sequence:
    st.info("Please upload a FASTA file to begin analysis.")
else:
    st.header(f"Analyzing: `{st.session_state.current_header}`")
    st.session_state.current_sequence = st.text_area(
        "Edit DNA Sequence (if needed):",
        value=st.session_state.current_sequence,
        height=150,
        key="dna_editor"
    )

    # --- Tabbed Interface ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "🧪 Basic Translation", "🔄 Reverse Complement", "🔍 ORF Finder",
        "⚙️ Batch Processing", "🧬 Sequence Alignment",
        "🧬 Secondary Structure (Conceptual)", "🧬 Primer Design", "✂️ Restriction Analysis",
        "📈 Sliding Window GC Content", "📈 Hydropathy Plot", "📄 Generate Report"
    ])

    with tab1:
        st.subheader("Basic Transcription & Translation")
        reading_frame = st.selectbox("Select Reading Frame", options=[1, 2, 3], index=0, key="rf_basic")
        if st.button("Analyze", key="analyze_basic"):
            if not validate_dna_sequence(st.session_state.current_sequence):
                st.error("Invalid DNA sequence. Please use only A, T, C, G.")
            else:
                with st.spinner("Translating..."):
                    try:
                        mrna_sequence = transcribe(st.session_state.current_sequence)
                        protein_details, protein_sequence = translate(mrna_sequence, reading_frame)
                        st.success("✅ Analysis Complete!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Original DNA (5' → 3')")
                            st.code(st.session_state.current_sequence, language='text')
                        with col2:
                            st.write("Transcribed mRNA (5' → 3')")
                            st.code(mrna_sequence, language='text')
                        protein_df = pd.DataFrame(protein_details)
                        st.dataframe(protein_df, width='stretch')
                        csv = protein_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Download Translation Details (CSV)", data=csv, file_name=f'translation_details_{st.session_state.current_header}.csv', mime='text/csv')
                        st.subheader("Final Protein Sequence")
                        st.write(f"Amino Acid Chain ({len(protein_sequence)} residues)")
                        st.code(protein_sequence, language='text')
                        if protein_sequence:
                            st.download_button(label="📥 Download Protein Sequence (TXT)", data=protein_sequence.encode('utf-8'), file_name=f'protein_{st.session_state.current_header}.txt', mime='text/plain')
                            aa_counts = pd.Series(list(protein_sequence)).value_counts().reset_index()
                            aa_counts.columns = ['Amino Acid', 'Count']
                            fig = px.bar(aa_counts, x='Amino Acid', y='Count', title='Amino Acid Composition', text='Count')
                            st.plotly_chart(fig, width='stretch')
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    with tab2:
        st.subheader("Analysis of Reverse Complement Strand")
        if st.button("Analyze Reverse Complement", key="analyze_rev_comp"):
            if not validate_dna_sequence(st.session_state.current_sequence):
                st.error("Invalid DNA sequence. Please use only A, T, C, G.")
            else:
                with st.spinner("Analyzing reverse strand..."):
                    try:
                        rev_comp_seq = reverse_complement(st.session_state.current_sequence)
                        mrna_rev = transcribe(rev_comp_seq)
                        st.write("Translation Results for Reverse Complement:")
                        for i in range(1, 4):
                            protein_details, protein_sequence = translate(mrna_rev, i)
                            if protein_sequence:
                                st.write(f"**Reading Frame +{i}:**")
                                st.code(protein_sequence, language='text')
                                st.dataframe(pd.DataFrame(protein_details), width='stretch')
                                st.divider()
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    with tab3:
        st.subheader("Open Reading Frame (ORF) Finder")
        min_len = st.slider("Minimum ORF Length (bp)", min_value=50, max_value=500, value=100, step=10)
        if st.button("Find ORFs", key="find_orfs"):
            if not validate_dna_sequence(st.session_state.current_sequence):
                st.error("Invalid DNA sequence. Please use only A, T, C, G.")
            else:
                with st.spinner("Scanning all 6 reading frames..."):
                    try:
                        orfs = find_orfs(st.session_state.current_sequence, min_len)
                        if orfs:
                            st.success(f"Found {len(orfs)} potential ORFs.")
                            orf_df = pd.DataFrame(orfs)
                            st.dataframe(orf_df, width='stretch')
                            csv = orf_df.to_csv(index=False).encode('utf-8')
                            st.download_button(label="📥 Download ORF Results (CSV)", data=csv, file_name=f'orfs_{st.session_state.current_header}.csv', mime='text/csv')
                        else:
                            st.warning("No ORFs found with the specified criteria.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    with tab4:
        st.subheader("Batch Translation of Multiple Sequences")
        if len(st.session_state.sequence_data) < 2:
            st.info("Please upload a FASTA file with at least 2 sequences to use this feature.")
        else:
            selected_headers = st.multiselect("Select sequences for batch analysis:", options=list(st.session_state.sequence_data.keys()), default=list(st.session_state.sequence_data.keys())[:3])
            if st.button("Run Batch Analysis", key="run_batch"):
                if not selected_headers:
                    st.warning("Please select at least one sequence.")
                else:
                    with st.spinner(f"Analyzing {len(selected_headers)} sequences..."):
                        for header in selected_headers:
                            seq = st.session_state.sequence_data[header]
                            with st.expander(f"Results for: {header}"):
                                try:
                                    mrna_seq = transcribe(seq)
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.write("**Reading Frame 1**")
                                        _, prot_seq = translate(mrna_seq, 1)
                                        st.code(prot_seq)
                                    with col2:
                                        st.write("**Reading Frame 2**")
                                        _, prot_seq = translate(mrna_seq, 2)
                                        st.code(prot_seq)
                                    with col3:
                                        st.write("**Reading Frame 3**")
                                        _, prot_seq = translate(mrna_seq, 3)
                                        st.code(prot_seq)
                                except Exception as e:
                                    st.error(f"Could not analyze {header}. Error: {e}")

    with tab5:
      st.subheader("Pairwise Global Sequence Alignment")
      if len(st.session_state.sequence_data) < 2:
          st.info("Please upload a FASTA file with at least 2 sequences to perform alignment.")
      else:
          col1, col2 = st.columns(2)
          with col1:
              seq1_header = st.selectbox("Select first sequence:", options=list(st.session_state.sequence_data.keys()))
          with col2:
              seq2_header = st.selectbox("Select second sequence:", options=list(st.session_state.sequence_data.keys()), index=1)

          if st.button("Align Sequences", key="run_alignment"):
              seq1 = st.session_state.sequence_data[seq1_header]
              seq2 = st.session_state.sequence_data[seq2_header]
              with st.spinner("Aligning..."):
                  try:
                      aligner = Align.PairwiseAligner()
                      aligner.mode = 'global'
                      aligner.match_score = 2
                      aligner.mismatch_score = -1
                      aligner.open_gap_score = -0.5
                      aligner.extend_gap_score = -0.1
                        
                      alignments = aligner.align(seq1, seq2)
                        
                      st.subheader("Best Alignment")
                      # The first alignment in the list is one of the best ones
                      best_alignment = alignments[0]
                      st.write(f"**Alignment Score:** {best_alignment.score}")
                        
                      # Display the formatted alignment string
                      st.code(str(best_alignment), language='text')
                        
                      st.write("**Alignment Statistics:**")
                      alignment_length = len(best_alignment.seqA)
                      st.write(f"- Length: {alignment_length}")
                        
                      # Calculate percentage identity based on the alignment length.
                      identity_percent = (best_alignment.identity / alignment_length) * 100 if alignment_length > 0 else 0
                      st.write(f"- Identity: {identity_percent:.2f}%")
                        
                  except Exception as e:
                      st.error(f"An unexpected error occurred during alignment: {e}")

    with tab6:
        st.subheader("Secondary Structure Prediction (Conceptual)")
        st.info("**Disclaimer:** Accurate protein secondary structure prediction is complex. This provides a basic analysis of amino acid properties, not a true structural prediction.")
        if st.button("Analyze Amino Acid Properties", key="analyze_properties"):
            if not st.session_state.current_sequence:
                st.error("No sequence loaded.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        mrna_sequence = transcribe(st.session_state.current_sequence)
                        protein_details, protein_sequence = translate(mrna_sequence, 1)
                        if not protein_sequence:
                            st.warning("No protein sequence could be generated for analysis.")
                        else:
                            hydrophobic_count = 0
                            hydrophilic_count = 0
                            for aa in protein_sequence:
                                if 'Hydrophobic' in AMINO_ACID_DETAILS.get(aa, {}).get('type', ''):
                                    hydrophobic_count += 1
                                elif 'Polar' in AMINO_ACID_DETAILS.get(aa, {}).get('type', '') or 'Charged' in AMINO_ACID_DETAILS.get(aa, {}).get('type', ''):
                                    hydrophilic_count += 1
                            total_residues = len(protein_sequence)
                            fig = px.pie(names=['Hydrophobic', 'Hydrophilic/Polar/Charged'], values=[hydrophobic_count, hydrophilic_count], title=f"Protein Property Distribution ({total_residues} residues)")
                            st.plotly_chart(fig, width='stretch')
                            st.write("**Interpretation:**")
                            st.write("- A high proportion of **hydrophobic** residues might suggest regions that are buried within the protein's core or could be part of a transmembrane domain.")
                            st.write("- A high proportion of **hydrophilic/polar/charged** residues is often found on the protein surface, interacting with the aqueous environment.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    with tab7:
        st.subheader("Primer Design Tool")
        st.info("Select a target region on the current sequence to design flanking primers.")
        seq_len = len(st.session_state.current_sequence)
        if seq_len < 50:
            st.warning("The sequence is too short for meaningful primer design. Please load a longer sequence (at least 50 bp).")
        elif seq_len == 0:
            st.warning("The sequence is empty. Please load a valid sequence.")
        else:
            default_start = max(1, int(seq_len * 0.2))
            default_end = max(default_start + 1, int(seq_len * 0.8))
            col1, col2 = st.columns(2)
            with col1:
                target_start = st.number_input("Target Start Position (1-based index):", min_value=1, max_value=seq_len, value=default_start, key="primer_start")
            with col2:
                target_end = st.number_input("Target End Position (1-based index):", min_value=1, max_value=seq_len, value=default_end, key="primer_end")
            primer_length = st.slider("Desired Primer Length (bp)", min_value=15, max_value=30, value=20, key="primer_len")
            if st.button("Design Primers", key="design_primers"):
                start_0_based = target_start - 1
                end_0_based = target_end - 1
                if start_0_based >= end_0_based:
                    st.error("Target Start Position must be less than Target End Position.")
                else:
                    with st.spinner("Designing primers..."):
                        try:
                            primers = design_primers(st.session_state.current_sequence, start_0_based, end_0_based, primer_length)
                            if "error" in primers:
                                st.error(primers["error"])
                            else:
                                st.success(f"Primers designed for target region {primers['target_region']}.")
                                fp = primers["forward_primer"]
                                rp = primers["reverse_primer"]
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Forward Primer", fp['sequence'])
                                    st.write(f"**Length:** {fp['length']} bp")
                                    st.write(f"**Tm:** {fp['tm']} °C")
                                    st.write(f"**GC Content:** {fp['gc']}%")
                                with col2:
                                    st.metric("Reverse Primer", rp['sequence'])
                                    st.write(f"**Length:** {rp['length']} bp")
                                    st.write(f"**Tm:** {rp['tm']} °C")
                                    st.write(f"**GC Content:** {rp['gc']}%")
                                primer_df = pd.DataFrame([
                                    {'Primer': 'Forward', 'Sequence': fp['sequence'], 'Length': fp['length'], 'Tm (C)': fp['tm'], 'GC Content (%)': fp['gc']},
                                    {'Primer': 'Reverse', 'Sequence': rp['sequence'], 'Length': rp['length'], 'Tm (C)': rp['tm'], 'GC Content (%)': rp['gc']}
                                ])
                                st.dataframe(primer_df, width='stretch')
                                csv = primer_df.to_csv(index=False).encode('utf-8')
                                st.download_button(label="📥 Download Primer Details (CSV)", data=csv, file_name=f'primers_{st.session_state.current_header}.csv', mime='text/csv')
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")

    with tab8:
        st.subheader("Restriction Enzyme Site Finder")
        st.info(f"Scanning the sequence for {len(RESTRICTION_ENZYMES)} common restriction enzymes.")
        if st.button("Analyze Restriction Sites", key="analyze_restriction"):
            if not validate_dna_sequence(st.session_state.current_sequence):
                st.error("Invalid DNA sequence. Please use only A, T, C, G.")
            else:
                with st.spinner("Scanning for restriction sites..."):
                    try:
                        sites = find_restriction_sites(st.session_state.current_sequence.upper(), RESTRICTION_ENZYMES)
                        if sites:
                            st.success(f"Found {len(sites)} restriction sites.")
                            sites_df = pd.DataFrame(sites)
                            st.dataframe(sites_df, width='stretch')
                            st.subheader("Cut Site Summary")
                            cut_counts = sites_df['Enzyme'].value_counts().reset_index()
                            cut_counts.columns = ['Enzyme', 'Number of Cuts']
                            fig = px.bar(cut_counts, x='Enzyme', y='Number of Cuts', title='Number of Cuts per Enzyme', text='Number of Cuts')
                            st.plotly_chart(fig, width='stretch')
                            csv = sites_df.to_csv(index=False).encode('utf-8')
                            st.download_button(label="📥 Download Restriction Sites (CSV)", data=csv, file_name=f'restriction_sites_{st.session_state.current_header}.csv', mime='text/csv')
                        else:
                            st.warning("No restriction sites found for the enzymes in our database.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    with tab9:
        st.subheader("Sliding Window GC Content Analysis")
        st.info("This tool plots the GC content along the sequence to identify regional variations.")
        seq_len = len(st.session_state.current_sequence)
        if seq_len == 0:
            st.warning("The sequence is empty. Please load a valid sequence.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                window_size = st.slider("Window Size (bp)", min_value=10, max_value=min(500, seq_len), value=100, step=10, key="gc_window_size")
            with col2:
                step_size = st.slider("Step Size (bp)", min_value=1, max_value=50, value=10, key="gc_step_size")
            if st.button("Analyze GC Content", key="analyze_gc_sliding"):
                with st.spinner("Analyzing..."):
                    try:
                        gc_df = calculate_sliding_gc(st.session_state.current_sequence.upper(), window_size, step_size)
                        if gc_df.empty:
                            st.warning("Could not perform analysis. Check window and step sizes.")
                        else:
                            overall_gc = calculate_sequence_stats(st.session_state.current_sequence)['GC Content (%)']
                            fig = px.line(gc_df, x='Position', y='GC Content (%)', title=f'Sliding Window GC Content (Window: {window_size} bp, Step: {step_size} bp)')
                            fig.add_hline(y=overall_gc, line_dash="dash", line_color="red", annotation_text=f"Overall GC: {overall_gc}%")
                            st.plotly_chart(fig, width='stretch')
                            st.dataframe(gc_df, width='stretch')
                            csv = gc_df.to_csv(index=False).encode('utf-8')
                            st.download_button(label="📥 Download GC Data (CSV)", data=csv, file_name=f'gc_content_{st.session_state.current_header}.csv', mime='text/csv')
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    with tab10:
        st.subheader("Kyte-Doolittle Hydropathy Plot")
        st.info("This plot predicts transmembrane regions and amphipathic helices based on amino acid hydropathy.")
        mrna_sequence = transcribe(st.session_state.current_sequence)
        _, protein_sequence = translate(mrna_sequence, 1)
        if not protein_sequence:
            st.warning("No protein sequence could be generated from the current DNA. Cannot perform hydropathy analysis.")
        else:
            st.write(f"Analyzing protein sequence: `{protein_sequence[:50]}...` ({len(protein_sequence)} residues)")
            window_size = st.slider("Select Window Size (residues)", min_value=5, max_value=25, value=19, step=2, help="Typical values are 9-21. Larger windows smooth the plot.")
            if st.button("Generate Hydropathy Plot", key="generate_hydropathy"):
                with st.spinner("Calculating hydropathy scores..."):
                    try:
                        hydropathy_df = calculate_hydropathy(protein_sequence, window_size)
                        if hydropathy_df.empty:
                            st.error("Could not generate plot. The protein sequence might be too short.")
                        else:
                            fig = px.line(hydropathy_df, x='Position', y='Hydropathy Score', title=f'Hydropathy Plot for {st.session_state.current_header}', labels={'Hydropathy Score': 'Average Hydropathy Score', 'Position': 'Amino Acid Position'})
                            fig.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Hydrophilic / Hydrophobic Boundary")
                            fig.update_yaxes(range=[min(KYTE_DOOLITTLE_SCALE.values()) - 1, max(KYTE_DOOLITTLE_SCALE.values()) + 1])
                            st.plotly_chart(fig, width='stretch')
                            st.write("**How to Interpret:**")
                            st.write("- **Positive scores** indicate **hydrophobic** (water-repelling) regions. Peaks above 1.6 often suggest potential transmembrane helices.")
                            st.write("- **Negative scores** indicate **hydrophilic** (water-attracting) regions, often found on the protein surface.")
                            st.dataframe(hydropathy_df, width='stretch')
                            csv = hydropathy_df.to_csv(index=False).encode('utf-8')
                            st.download_button(label="📥 Download Hydropathy Data (CSV)", data=csv, file_name=f'hydropathy_{st.session_state.current_header}.csv', mime='text/csv')
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    with tab11:
        st.subheader("Generate Analysis Report")
        st.info("Select the analyses you want to include in a single, downloadable HTML report.")
        
        if 'report_options' not in st.session_state:
            st.session_state.report_options = {
                "stats": True, "translation": True, "aa_composition": True, "orfs": False, "primers": False, "restriction": False
            }

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.report_options["stats"] = st.checkbox("Sequence Statistics", value=st.session_state.report_options["stats"])
            st.session_state.report_options["translation"] = st.checkbox("Translation Details (RF1)", value=st.session_state.report_options["translation"])
            st.session_state.report_options["aa_composition"] = st.checkbox("Amino Acid Composition Plot", value=st.session_state.report_options["aa_composition"])
            st.session_state.report_options["orfs"] = st.checkbox("ORF Finder Results", value=st.session_state.report_options["orfs"])
        with col2:
            st.session_state.report_options["primers"] = st.checkbox("Primer Design (Default Settings)", value=st.session_state.report_options["primers"])
            st.session_state.report_options["restriction"] = st.checkbox("Restriction Site Analysis", value=st.session_state.report_options["restriction"])
        
        if st.button("📄 Generate HTML Report", key="generate_report"):
            with st.spinner("Generating report..."):
                try:
                    # --- Gather all data ---
                    report_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Analysis Report for {st.session_state.current_header}</title>
                        <style>
                            body {{ font-family: sans-serif; line-height: 1.6; margin: 2em; }}
                            h1, h2, h3 {{ color: #2c3e50; }}
                            table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #f2f2f2; }}
                            img {{ max-width: 100%; height: auto; }}
                            .metric {{ font-size: 1.2em; font-weight: bold; color: #3498db; }}
                        </style>
                    </head>
                    <body>
                        <h1>Analysis Report: {st.session_state.current_header}</h1>
                        <p><strong>Generated on:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    """
                    
                    # Sequence Stats
                    if st.session_state.report_options["stats"]:
                        stats = calculate_sequence_stats(st.session_state.current_sequence)
                        report_html += f"""
                        <h2>Sequence Statistics</h2>
                        <ul>
                            <li><strong>Length:</strong> {stats['Length']} bp</li>
                            <li><strong>GC Content:</strong> {stats['GC Content (%)']}%</li>
                            <li><strong>A Count:</strong> {stats['A Count']}</li>
                            <li><strong>T Count:</strong> {stats['T Count']}</li>
                            <li><strong>C Count:</strong> {stats['C Count']}</li>
                            <li><strong>G Count:</strong> {stats['G Count']}</li>
                        </ul>
                        """
                    
                    # Translation
                    if st.session_state.report_options["translation"]:
                        mrna = transcribe(st.session_state.current_sequence)
                        protein_details, protein_seq = translate(mrna, 1)
                        report_html += f"""
                        <h2>Translation Details (Reading Frame 1)</h2>
                        <p><strong>Protein Sequence ({len(protein_seq)} residues):</strong></p>
                        <code>{protein_seq}</code>
                        {pd.DataFrame(protein_details).to_html(index=False, escape=False)}
                        """
                    
                    # AA Composition
                    if st.session_state.report_options["aa_composition"] and protein_seq:
                        aa_counts = pd.Series(list(protein_seq)).value_counts().reset_index()
                        aa_counts.columns = ['Amino Acid', 'Count']
                        fig = px.bar(aa_counts, x='Amino Acid', y='Count', title='Amino Acid Composition', text='Count')
                        img_tag = f'<img src="{fig_to_base64(fig)}" alt="Amino Acid Composition Plot">'
                        report_html += f"<h2>Amino Acid Composition</h2>{img_tag}"

                    # ORFs
                    if st.session_state.report_options["orfs"]:
                        orfs = find_orfs(st.session_state.current_sequence)
                        if orfs:
                            report_html += f"<h2>Open Reading Frames (ORFs)</h2>{pd.DataFrame(orfs).to_html(index=False, escape=False)}"
                        else:
                            report_html += "<h2>Open Reading Frames (ORFs)</h2><p>No ORFs found with default settings.</p>"

                    # Primers
                    if st.session_state.report_options["primers"]:
                        seq_len = len(st.session_state.current_sequence)
                        if seq_len > 50:
                            default_start = max(1, int(seq_len * 0.2))
                            default_end = max(default_start + 1, int(seq_len * 0.8))
                            primers = design_primers(st.session_state.current_sequence, default_start - 1, default_end - 1)
                            if "error" not in primers:
                                fp, rp = primers["forward_primer"], primers["reverse_primer"]
                                primer_df = pd.DataFrame([
                                    {'Primer': 'Forward', 'Sequence': fp['sequence'], 'Length (bp)': fp['length'], 'Tm (C)': fp['tm'], 'GC Content (%)': fp['gc']},
                                    {'Primer': 'Reverse', 'Sequence': rp['sequence'], 'Length (bp)': rp['length'], 'Tm (C)': rp['tm'], 'GC Content (%)': rp['gc']}
                                ])
                                report_html += f"<h2>Primer Design (Default Settings)</h2>{primer_df.to_html(index=False, escape=False)}"
                    
                    # Restriction Sites
                    if st.session_state.report_options["restriction"]:
                        sites = find_restriction_sites(st.session_state.current_sequence.upper(), RESTRICTION_ENZYMES)
                        if sites:
                            report_html += f"<h2>Restriction Site Analysis</h2>{pd.DataFrame(sites).to_html(index=False, escape=False)}"
                        else:
                            report_html += "<h2>Restriction Site Analysis</h2><p>No restriction sites found for enzymes in our database.</p>"
                    
                    report_html += "</body></html>"
                    
                    st.success("Report generated successfully!")
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=report_html,
                        file_name=f'report_{st.session_state.current_header}.html',
                        mime='text/html'
                    )

                except Exception as e:
                    st.error(f"An unexpected error occurred during report generation: {e}")


# --- Educational Content (unchanged) ---
st.divider()
with st.expander("📚 The Science Behind It"):
    st.markdown("""
    #### **Transcription (DNA → mRNA)**
    This process occurs in the cell nucleus. The enzyme RNA polymerase reads the DNA template strand and creates a complementary messenger RNA (mRNA) strand.
    - **Adenine (A)** pairs with **Uracil (U)** in RNA.
    - **Thymine (T)** pairs with **Adenine (A)**.
    - **Cytosine (C)** pairs with **Guanine (G)**.
    - **Guanine (G)** pairs with **Cytosine (C)**.

    #### **Translation (mRNA → Protein)**
    The mRNA strand travels to a ribosome, where it is read three nucleotides at a time, known as a **codon**. Each codon specifies a particular **amino acid**. The **reading frame** determines which three-base group is the first codon. Transfer RNA (tRNA) molecules bring the correct amino acids to the ribosome, which are then linked together to form a polypeptide chain (a protein). Translation stops when a "stop" codon is encountered.
    
    #### **Reverse Complement**
    DNA is double-stranded and antiparallel. The reverse complement is the sequence of the opposite strand, read in the opposite direction. It is essential because genes can be located on either strand.
    
    #### **Open Reading Frame (ORF)**
    An ORF is a stretch of codons that starts with a start codon (ATG) and ends with a stop codon (TAA, TAG, or TGA). Finding ORFs is a primary method for identifying potential genes within a long DNA sequence.
    
    #### **Sequence Alignment**
    Alignment is the process of arranging two or more sequences to identify regions of similarity. These similarities may be a consequence of functional, structural, or evolutionary relationships between the sequences. The score represents the quality of the alignment based on matches, mismatches, and gaps.
    
    #### **Restriction Enzymes**
    These are enzymes that cut DNA at specific, short sequences (recognition sites). They are fundamental tools in molecular biology for genetic engineering and cloning.
    
    #### **Hydropathy**
    A measure of the hydrophobic or hydrophilic character of a molecule. In proteins, a hydropathy plot can help predict transmembrane regions, which are typically composed of long stretches of hydrophobic amino acids.
    """)