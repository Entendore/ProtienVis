import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Genetic code dictionary for translation
genetic_code = {
    'AUG':'Met', 'UUU':'Phe', 'UUC':'Phe', 'UUA':'Leu', 'UUG':'Leu',
    'CUU':'Leu', 'CUC':'Leu', 'CUA':'Leu', 'CUG':'Leu',
    'AUU':'Ile', 'AUC':'Ile', 'AUA':'Ile',
    'GUU':'Val', 'GUC':'Val', 'GUA':'Val', 'GUG':'Val',
    'UCU':'Ser', 'UCC':'Ser', 'UCA':'Ser', 'UCG':'Ser',
    'CCU':'Pro', 'CCC':'Pro', 'CCA':'Pro', 'CCG':'Pro',
    'ACU':'Thr', 'ACC':'Thr', 'ACA':'Thr', 'ACG':'Thr',
    'GCU':'Ala', 'GCC':'Ala', 'GCA':'Ala', 'GCG':'Ala',
    'UAU':'Tyr', 'UAC':'Tyr', 'UAA':'STOP', 'UAG':'STOP',
    'CAU':'His', 'CAC':'His', 'CAA':'Gln', 'CAG':'Gln',
    'AAU':'Asn', 'AAC':'Asn', 'AAA':'Lys', 'AAG':'Lys',
    'GAU':'Asp', 'GAC':'Asp', 'GAA':'Glu', 'GAG':'Glu',
    'UGU':'Cys', 'UGC':'Cys', 'UGA':'STOP', 'UGG':'Trp',
    'CGU':'Arg', 'CGC':'Arg', 'CGA':'Arg', 'CGG':'Arg',
    'AGU':'Ser', 'AGC':'Ser', 'AGA':'Arg', 'AGG':'Arg',
    'GGU':'Gly', 'GGC':'Gly', 'GGA':'Gly', 'GGG':'Gly'
}

# Example DNA template strand
dna_template = "TACGTTGACGTAACGACT"  # encodes Met-Val-Asp-Val-Thr

# Transcription: DNA -> mRNA
transcription_map = {"A":"U", "T":"A", "C":"G", "G":"C"}
mrna = "".join([transcription_map[base] for base in dna_template])

# Translation: mRNA -> Protein
codons = [mrna[i:i+3] for i in range(0, len(mrna), 3)]
protein = []
for codon in codons:
    aa = genetic_code.get(codon, "")
    if aa == "STOP":
        break
    protein.append(aa)

# Visualization setup
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

dna_text = ax.text(0.05, 0.9, f"DNA: {dna_template}", fontsize=12, family="monospace")
mrna_text = ax.text(0.05, 0.7, "mRNA: ", fontsize=12, family="monospace")
protein_text = ax.text(0.05, 0.5, "Protein: ", fontsize=12)

step = {"i": 0}

def update(frame):
    i = step["i"]

    # First phase: transcription
    if i < len(mrna):
        mrna_text.set_text("mRNA: " + mrna[:i+1])
    # Second phase: translation (codon by codon)
    elif i < len(mrna) + len(protein):
        j = i - len(mrna)
        protein_text.set_text("Protein: " + "-".join(protein[:j+1]))

    step["i"] += 1
    return mrna_text, protein_text

ani = animation.FuncAnimation(fig, update, frames=len(mrna)+len(protein)+2,
                              interval=600, blit=False, repeat=False)

plt.show()
