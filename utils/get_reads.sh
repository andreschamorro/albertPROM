#!/bin/sh

igsr_df()
{
	wget -a wget.log -nv -O - ${1} | bcftools view -i'ALT="<INS:ME:LINE1>"' > variants/$(basename "${1/genotypes/LINE1}");
}

get_rate()
{
	echo $(bc -l <<< $(samtools view -c $2)/$(samtools view -c $1));
}

pa_reads()
{
	rate=$(get_rate $1 $2)
	samtools fastq -@16 -o reads/LINE1/$(basename -s .bam $1).LINE1_R.fq -0 /dev/null -n $1
	samtools view -h --subsample $rate | samtools fastq -@2 -o reads/LINE1/$(basename -s .bam $2)._R.fq -0 /dev/null -n $1
}


# call arguments verbatim:
"$@"
