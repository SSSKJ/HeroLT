#!/bin/bash

dataset=$1

function gdrive-get() {
    fileid=$1
    filename=$2
    if [[ "${fileid}" == "" || "${filename}" == "" ]]; then
        echo "gdrive-curl gdrive-url|gdrive-fileid filename"
        return 1
    else
        if [[ ${fileid} = http* ]]; then
            fileid=$(echo ${fileid} | sed "s/http.*drive.google.com.*id=\([^&]*\).*/\1/")
        fi
        echo "Download ${filename} from google drive with id ${fileid}..."
        cookie="/tmp/cookies.txt"
        curl -c ${cookie} -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
        confirmid=$(awk '/download/ {print $NF}' ${cookie})
        curl -Lb ${cookie} "https://drive.google.com/uc?export=download&confirm=${confirmid}&id=${fileid}" -o ${filename}
        rm -rf ${cookie}
        return 0
    fi
}

mkdir -p ${dataset}
cd ${dataset}



if [ ${dataset} == 'eurlex-4k' ]; then
    gdrive-get 1ZplgIYqJavtAJNaqnyvaE77IZZbCrXJw indexer.tar.gz
	gdrive-get 1Kv8_62RU1gfiAULCPvkMY5kk1ttTxQzc proc_data.tar.gz
    gdrive-get 1sxiwozg5hCDc28BuNyYt8R3Vm_TbIJXI pifa-tfidf-s0.bert.tar.gz
    gdrive-get 19SO8koS8mCapG0wM3dF_ffVZeMpKMuCJ pifa-tfidf-s0.roberta.tar.gz
    gdrive-get 1R8gXkvG8lwFawJYHthNZn3kX9XyzC82g pifa-tfidf-s0.xlnet.tar.gz
    gdrive-get 1cZ4JtRuCTucvrjwLBjFADfosc2ugxrAG pifa-neural-s0.bert.tar.gz
    gdrive-get 1gTJoNtJc6VdY9B2pbyLANr3y3QWrn_4J pifa-neural-s0.roberta.tar.gz
    gdrive-get 181G_Oi51Cqp9-aGKhtcp6r6VaHpltNSM pifa-neural-s0.xlnet.tar.gz
    gdrive-get 1TD3fvN0YLuJwl4t2Tgl1Ms7ViJaU1Faj text-emb-s0.bert.tar.gz
    gdrive-get 1i-gyrDglBwjgpbigITz-0C7uEcO1EPQ9 text-emb-s0.roberta.tar.gz
    gdrive-get 14OrxieNLivGFK-rFEmubsb8JEh5kzQjY text-emb-s0.xlnet.tar.gz

elif [ ${dataset} == 'wiki10-31k' ]; then
	gdrive-get 1cQvX5ayDGwgGc4hCM9XHAVRISWaSknFC indexer.tar.gz
	gdrive-get 1A87V4CaY-PwuiqrB61z5_2WWpFWFt_db proc_data.tar.gz
    gdrive-get 1wujzxiUePpEo0hoZYc1QqczFzDngR22y pifa-tfidf-s0.bert.tar.gz
    gdrive-get 1uYHml0FlvXfWy8wAy7fiZQGPlPQkPThE pifa-tfidf-s0.roberta.tar.gz
    gdrive-get 1-GBDHry_ThSWN_FYJe4swSvuBOmFBYaJ pifa-tfidf-s0.xlnet.tar.gz
    gdrive-get 1_MnCXY3DiS2Pi8OCgPAxt98LyEpUGsMR pifa-neural-s0.bert.tar.gz
    gdrive-get 1TXEg37XNtNFa8DbB6YwRy9qgfW9Od1x5 pifa-neural-s0.roberta.tar.gz
    gdrive-get 1_pZVYYT_8H2sf1N0PEd5kxgzskMX43U9 pifa-neural-s0.xlnet.tar.gz
    gdrive-get 1GecTrAlwAvFZl5cUIxy388x49hTO8WAL text-emb-s0.bert.tar.gz
    gdrive-get 1_L3AMw5uGadSTVyAbWAD25SuXM9F3AgZ text-emb-s0.roberta.tar.gz
    gdrive-get 1ZbXP-wTsEhlcl67XsRgCTXgg5bVuBJI6 text-emb-s0.xlnet.tar.gz

elif [ ${dataset} == 'amazoncat-13k' ]; then
	gdrive-get 15h1l05M3zxyjQQQQLz7mNoll5SmTl9B_ indexer.tar.gz
	gdrive-get 1VN39DazmTb3GvdO5qDNp15BEhd40c4HY proc_data.tar.gz
    gdrive-get 1cUfCRcgcoeu-DW6r1D22D88sUx_5MNU6 pifa-tfidf-s0.bert.tar.gz
    gdrive-get 1hNcG-URVZ6Dx3z5LNwOJ1xq-Pgx-4_Cs pifa-tfidf-s0.roberta.tar.gz
    gdrive-get 1wK0wTrrazAAaYguI0QKPhfDGBap-bO9q pifa-tfidf-s0.xlnet.tar.gz
    gdrive-get 123NxkH9Sw0IDpEKdHgjnfWbniTEE96n7 pifa-neural-s0.bert.tar.gz
    gdrive-get 1MWcIPlLlPeIPQ7IYS64-b0G56OW3hxFk pifa-neural-s0.roberta.tar.gz
    gdrive-get 1zkRWN-IVUeF2wImrroH0s1dPPOlnIqNa pifa-neural-s0.xlnet.tar.gz
    gdrive-get 1fbgGIOlYF4lWLqr86SC72eLcAvYb9sMw text-emb-s0.bert.tar.gz
    gdrive-get 1VnFGL15WbyyqAEOgHbPlBRWAcmCz-L50 text-emb-s0.roberta.tar.gz
    gdrive-get 1vnbL1wUGiYeLfZ-y5w1q9nljn3sGUlxD text-emb-s0.xlnet.tar.gz

else
	echo "unknown dataset [ eurlex-4k | wiki10-31k | amazonCat-13k ]"
	exit
fi


tar -xzvf indexer.tar.gz
tar -xzvf proc_data.tar.gz

label_emb_arr=( pifa-tfidf pifa-neural text-emb )
model_type_arr=( bert roberta xlnet )
for label_emb in "${label_emb_arr[@]}"; do
    for model_type in "${model_type_arr[@]}"; do
        tar -xzvf ${label_emb}-s0.${model_type}.tar.gz
    done
done