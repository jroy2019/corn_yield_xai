# Genotype Data Processing Workflow

## **Workflow**
1. **Filtering SNPs** (removing SNPs with low Minor Allele Frequency)
2. **Linkage Disequilibrium Pruning** (removing highly correlated SNPs)
3. **Phasing and Imputation** (estimating missing genotypes)
4. **Numerical Encoding of SNPs**

## **Dependencies**
- [PLINK v1.9](https://www.cog-genomics.org/plink/) 
- [VCFtools](https://vcftools.github.io/)
- [Beagle 5.4](https://faculty.washington.edu/browning/beagle/beagle.html)

Input: 5_Genotype_Data_All_2014_2025_Hybrids.vcf file in the Genome 2 Fields 2024 Training Data.

## 1. Filtering SNPs with Minor Allele Frequency (MAF) < 0.01

### **Overview**
Filters out SNPs with a **Minor Allele Frequency (MAF) < 0.01** using VCFtools.

 
```bash
vcftools --vcf 5_Genotype_Data_All_2014_2025_Hybrids.vcf --maf 0.01 --recode --out filtered_g2f_SNPs.recode.vcf
```

### Results:
- 39 SNPs removed.
- 2386 SNP sites out of 2425 SNPs were retained.
- All 5899 hybrids retained.

## 2. Linkage Disequilibrium (LD) Pruning

### **Overview**
Performs **LD pruning** to remove SNPs in linkage disequilibrium (r^2 > 0.7 in sliding window of 50 SNPs). <br>

Pruning involves:
- Cleaning the VCF file
- Sorting SNPs by chromosome and position
- Using PLINK to perform LD pruning

#### **Step 1: Ensure uniform formating of genotypes**
```bash
perl -pi -e 's/\t0\t/\t0\/\.\t/g; s/\t1\t/\t1\/\.\t/g; s/\t\.\t/\t\.\/\.\t/g; s/\t0\n/\t0\/\.\n/g; s/\t1\n/\t1\/\.\n/g; s/\t\.\n/\t\.\/\.\n/g' filtered_g2f_SNPs.recode.vcf
```

#### **Step 2: Order SNPs based on chromsome number and position**
```bash
for chr in {1..10}; do
    grep '^#' filtered_g2f_SNPs.recode.vcf > chrom${chr}_filtered_SNPs.recode.vcf
    awk -v c="$chr" '$1==c' filtered_g2f_SNPs.recode.vcf | sort -t$'\t' -k2,2n >> chrom${chr}_filtered_SNPs.recode.vcf
done
```

#### **Step 3: Combine separate  vcf files into 1 vcf file**

```bash
cat chrom*_filtered_SNPs.recode.vcf >> combined_filtered_chromosomes.recode.vcf
```

#### **Step 4: Remove excess headers in combined vcf file. **
```bash
head -n9 combined_filtered_chromosomes.recode.vcf > v2_combined_filtered_chromosomes.recode.vcf
grep -v "#" combined_filtered_chromosomes.recode.vcf >> v2_combined_filtered_chromosomes.recode.vcf
```

#### **Step 5: Perform LD Pruning of SNPs**
```bash
/software/projects/pawsey0149/jroy/plink --indep-pairwise 50 5 0.7 \
    --out pruned_combined_filtered_chromosomes.recode.vcf \
    --r2 gz --vcf v2_combined_filtered_chromosomes.recode.vcf \
    --allow-extra-chr --threads 8 --double-id
```
Hyperparameters: sliding window=50, step=5 SNPs, minimum LD pruned in = 0.7 

```bash
unzip pruned_combined_filtered_chromosomes.recode.vcf.ld.gz
```

#### **Step 6: Extract pruned-in snps only**
```bash
/software/projects/pawsey0149/jroy/plink -vcf v2_combined_filtered_chromosomes.recode.vcf --extract pruned_combined_filtered_chromosomes.recode.vcf.prune.in --make-bed --out extracted_pruned_SNPs.recode.vcf --allow-extra-chr --threads 8 --double-id
```

```bash
/software/projects/pawsey0149/jroy/plink --bfile extracted_pruned_SNPs.recode.vcf \
    --recode vcf-iid --allow-extra-chr --threads 8 --double-id \
    --out extract_pruned_filtered_SNPs.recode.vcf
```

### Results:
- Pruned out 233 of 2386 variants.
- 2153 of 2386 variants retained.
- All 5899 hybrids retained.

## 3. Phasing & Imputation of SNP Data
 
```bash
/software/projects/pawsey0149/jroy/beagle.17Dec24.224.jar --vcf extract_pruned_filtered_SNPs.recode.vcf \
    --allow-extra-chr out=/scratch/pawsey0149/jroy/final_extracted_pruned_phased_filtered_SNPs.recode.vcf
```

### Results:
- SNPs imputed per chromosome: 
  - Chromosome 1: 367
  - Chromosome 2: 333
  - Chromosome 3: 246
  - Chromosome 4: 215
  - Chromosome 5: 315
  - Chromosome 6: 158
  - Chromosome 7: 201
  - Chromosome 8: 261
  - Chromosome 9: 168
  - Chromosome 10: 122
- All 5899 individuals retained.

## 4. Numerical Encoding and Formating of SNP Data for Machine Learning

Remove all metadata (lines starting with ##) except column names. I did this manually 

#### **Step 1: Remove unnecessary columns**
```bash
cut -f 3,10- final_extracted_pruned_filtered_SNPs.recode.vcf > id_hybrids_info_only.vcf
```

#### **Step 2: Numerical Encoding**
```bash
sed -i -E 's/\b0\|0\b/0/g; s/\b0\|1\b/0.5/g; s/\b1\|0\b/0.5/g; s/\b1\|1\b/1/g' id_hybrids_info_only.vcf
```

#### **Step 3: Transpose Data** 
```bash
awk '{
    for (i=1; i<=NF; i++) {
        matrix[i, NR] = $i;
    }
    max_cols = NF;
    max_rows = NR;
} 
END {
    for (i=1; i<=max_cols; i++) {
        for (j=1; j<=max_rows; j++) {
            printf "%s%s", matrix[i, j], (j==max_rows ? "\n" : "\t");
        }
    }
}' id_hybrids_info_only.vcf > numeric_formated_hyrid_info.txt
```







