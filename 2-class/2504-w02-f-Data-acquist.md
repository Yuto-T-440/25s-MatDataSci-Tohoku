---
title: "MatInformatics: Class 02b"
subtitle: "Profs: Pawan K. Tripathi"
author: ".."
date: "`{DATE}`"  #  You'll need to replace this with Python code.
output:
  pdf_document:
    latex_engine: xelatex
    toc: true
    number_sections: true
    toc_depth: 6
    highlight: tango
    extra_options: " -F --filter pandoc-unicode "
  html_document:
    css: ../lab.css
    highlight: pygments
    theme: cerulean
    toc: true
    toc_depth: 6
    toc_float: true
    df_print: paged
urlcolor: blue
always_allow_html: true
header-includes:
  - "\\usepackage{placeins}"
  - "\\usepackage[utf8]{inputenc}"
---

# Week 2: Data Acquisition and Preparation 
## **Pawan K. Tripathi**
Research Assistant Professor
Case Western Reserve University, Cleveland, Ohio, USA


* **Format:** 6 remote sessions (90 mins each) via [Zoom]
* **Tools:** [Google Colab](https://colab.research.google.com/#scrollTo=-Rh3-Vt9Nev9) (free, web-based Python environment)

* **Communication:** [Slack, Email] for questions & announcements
* **Materials:** Slides & Colab notebooks provided before class
* **Expectations:** Active participation, willingness to learn basic Python/data concepts.

---

#  Representing Materials Data for Computers

**Turning Physical Materials into Digital Information**

---

## Today's Agenda 

1.  **Recap & Q&A**  - What is MI? Why do we need it?
2.  **Types of Materials Data**  - Composition, Structure, Processing, Properties.
3.  **Features & Descriptors** - Making data machine-readable.
4.  **Hands-on Demo Outline (Colab)**  - Loading data, basic featurization.
5.  **Q&A & Look Ahead** 

---

## Recap: Class 1 Key Ideas

* **Materials Science:** Aims to find/design materials with desired **properties** (strength, conductivity, etc.).
* **Data Science:** Uses tools to find patterns, build models, and make predictions from **data**.
* **Materials Informatics (MI):** Combines both to **accelerate** materials discovery and design using data-driven approaches.
* **Simple Workflow:** Acquire Data -> Represent Data -> Build Model -> Predict -> Validate.

**Any questions from last time?**

---

## Part 1: Types of Materials Data

How do we describe a material?

1.  **Composition:** *What* elements are in it, and how much of each?
    * Examples: $H_2O$, $NaCl$, $Fe_{0.8}Ni_{0.2}$, $Al_2O_3$
    * Digital Representation:
        * String: `"Fe2O3"`
        * Dictionary/Map: `{'Fe': 2, 'O': 3}`
        * List of elements + fractions: `(['Al', 'O'], [0.4, 0.6])`

2.  **Structure:** *How* are the atoms arranged in space?
    * Crucial! Graphite vs. Diamond (both Carbon, different structure -> different properties).
    * Concepts: Crystal lattice, unit cell, atomic coordinates. (We won't dive deep into crystallography).
    * Digital Representation: Standard file formats (CIF, POSCAR), lists of atomic positions, symmetry information (space group).
    * *Key Idea:* Structure determines many properties.

    ![Comparison of crystalline (ordered) vs amorphous (disordered) atomic structures](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Crystalline_vs_Amorphous.svg/500px-Crystalline_vs_Amorphous.svg.png)
    *(Caption: Atomic arrangement matters: Ordered (crystalline) vs. Disordered (amorphous))*
   
---

## Types of Materials Data (Continued)

3.  **Processing:** *How* was the material made?
    * Examples: Temperature, pressure, cooling rate, synthesis method (e.g., casting, sputtering, 3D printing).
    * Digital Representation: Numerical values (e.g., `Temperature: 1200 K`), categorical labels (e.g., `Method: 'Sol-Gel'`).
    * *Key Idea:* Processing influences the final structure and properties.

4.  **Properties:** *What* can the material do? (Often the target of our predictions!)
    * **Mechanical:** Hardness, Strength, Elasticity (Young's Modulus)
    * **Electronic:** Conductivity (Metal/Insulator), Band Gap (Semiconductors)
    * **Thermal:** Melting Point, Thermal Conductivity
    * **Magnetic:** Magnetization
    * **Optical:** Color, Refractive Index
    * Digital Representation: Numerical values (e.g., `Band Gap: 1.1 eV`), categorical (e.g., `Magnetic: True`).

---

## Part 2: Features & Descriptors - Making Data Machine-Readable

* Computers and ML models need **numerical input**. We can't just feed them "NaCl".
* **Featurization:** The process of converting raw material information (like composition or structure) into a vector (list) of numerical values called **features** or **descriptors**.
* **Goal:** Create a table (like a spreadsheet or `pandas` DataFrame) where:
    * Rows = Materials
    * Columns = Numerical Features

    ![Illustration of a data table with rows for samples and columns for features](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Feature_vector_illustration.svg/600px-Feature_vector_illustration.svg.png)
    *(Caption: Transforming raw data into a structured feature table)*
 

    | Material | Feature 1 (e.g., Avg Atomic #) | Feature 2 (e.g., Avg Electronegativity) | ... | Target Property |
    | :------- | :----------------------------- | :-------------------------------------- | :-: | :-------------- |
    | NaCl     | 14.0                           | 2.1                                     | ... | 1074 K (Melt Pt)|
    | SiC      | 10.0                           | 2.2                                     | ... | 3103 K (Melt Pt)|
    | ...      | ...                            | ...                                     | ... | ...             |

---

## Simple Compositional Features

* Features derived *only* from the chemical formula (e.g., $Al_2O_3$).
* Based on properties of the constituent elements (found in periodic table data).
* **Examples:**
    * Average atomic number
    * Average atomic weight
    * Sum of atomic weights
    * Range/Difference in electronegativity between elements
    * Average number of valence electrons
    * Fraction of metallic elements

* **Why?** These simple numbers capture some essence of the chemistry without needing complex structural info. Easy to calculate!

    ![Periodic Table of Elements](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Periodic_table_large.svg/800px-Periodic_table_large.svg.png)
    *(Caption: Elemental properties from the periodic table form the basis for compositional features)*
  

---

## Structural Features (Conceptual)

* Features derived from the 3D arrangement of atoms.
* Often more powerful predictors, but require structural data (which isn't always available).
* **Examples (High Level):**
    * Density (Mass / Volume)
    * Unit cell volume
    * Space group (A classification of crystal symmetry - treated as a category)
    * Coordination numbers (How many neighbors each atom has)
    * Bond lengths and angles
    * More complex graph-based or topological descriptors

* *We will focus more on compositional features in this intro course for simplicity.*

---

## Part 3: Hands-on Demo Outline (Google Colab)

* **Goal:** Load a simple materials dataset and generate basic compositional features.
* **Tools:** `pandas` (for data tables), maybe `pymatgen` or `matminer` (for materials-specific tools).

* **Steps:**
    1.  **Load Data:** Read a CSV file containing material compositions (e.g., formulas like "NaCl", "Fe2O3") and maybe a target property (e.g., melting point).
        ```python
        # import pandas as pd
        # df = pd.read_csv('simple_materials.csv')
        # print(df.head())
        ```
    2.  **Represent Composition:** Show how `pandas` stores the formula string. Discuss converting it to a more structured format if needed (e.g., using `pymatgen.Composition`).
        ```python
        # from pymatgen.core import Composition
        # comp_obj = Composition("Fe2O3")
        # print(comp_obj.get_atomic_fraction("Fe"))
        ```
    3.  **Featurization (Simple Example):**
        * Calculate a simple feature manually using element data (e.g., average atomic mass). Need a dictionary or source for element properties.
        * *OR* Introduce a library function (like `matminer.featurizers.composition.ElementProperty.from_preset("magpie")`) to automatically generate many features. Show the input (list of compositions) and the output (DataFrame of features).
        ```python
        # Example using a hypothetical function
        # features = calculate_simple_features(df['composition_column'])
        # print(features.head())
        ```
    4.  **Combine Data:** Merge the generated features back with the original DataFrame. Show the final table ready for ML.
        ```python
        # final_df = pd.concat([df, features], axis=1)
        # print(final_df.head())
        ```

---

## Part 4: Q&A & Look Ahead

* **Any Questions?**

* **Think about:**
    * Why do we need different types of features (compositional vs. structural)?
    * How might the *way* we represent data (the features we choose) affect the performance of a machine learning model later?
    * Consider a recipe: What are its "composition", "processing", "structure" (maybe presentation?), and "properties" (taste, texture)? How would you featurize a recipe?

* **Next Time:** Finding and Wrangling Materials Data - Where does this data come from, and why is it often messy?

---

**Thank You!**
