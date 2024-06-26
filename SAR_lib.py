# TRABAJO REALIZADO POR:
# Castelló Beltrán, J; Guzman Alamos, R & Jaen Ruiz, A.J.
# MÉTODOS IMPLEMENTADOS:
#   index_file
#   show_stats
#   solve_query
#   get_posting
#   get_positionals
#   reverse_posting
#   and_posting
#   or_posting
#   solve_and_show


import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
import sys
import math
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia

        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [
        ("all", True),
        ("title", True),
        ("summary", True),
        ("section-name", True),
        ("url", False),
    ]
    def_field = "all"
    PAR_MARK = "%"
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    all_atribs = [
        "urls",
        "index",
        "sindex",
        "ptindex",
        "docs",
        "weight",
        "articles",
        "tokenizer",
        "stemmer",
        "show_all",
        "use_stemming",
    ]

    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas

        """
        self.urls = set()  # hash para las urls procesadas,
        self.index = (
            {}
        )  # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.sindex = (
            {}
        )  # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {}  # hash para el indice permuterm.
        self.docs = (
            {}
        )  # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {}  # hash de terminos para el pesado, ranking de resultados.
        self.articles = (
            {}
        )  # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile(
            "\W+"
        )  # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer("spanish")  # stemmer en castellano
        self.show_all = False  # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False  # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = (
            False  # valor por defecto, se cambia con self.set_stemming()
        )
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()

        ##Utiles para la tokenización de los textos
        self.r1 = re.compile("[.;?!]")
        self.r2 = re.compile("\W+")
        # Para eliminar los simbolos no alfanuméricos
        self.r3 = re.compile("[^a-zA-Z0-9\s]")
        self.info = {}
       

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################

    def set_showall(self, v: bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v

    def set_snippet(self, v: bool):
        """

        Cambia el modo de mostrar snippet.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v

    def set_stemming(self, v: bool):
        """

        Cambia el modo de stemming por defecto.

        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v

    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################

    def save_info(self, filename: str):
        """
        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, "wb") as fh:
            pickle.dump(info, fh)

    def load_info(self, filename: str):
        """
        Carga la información del índice desde un fichero en formato binario

        """
        # info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, "rb") as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article: Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article["url"] in self.urls

    def index_dir(self, root: str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.multifield = args["multifield"]
        self.positional = args["positional"]
        self.stemming = args["stem"]
        self.permuterm = args["permuterm"]

        # id de cada documento:
        id = 0
       
        for field, tokenize in self.fields:
            self.index[field] = {}
        
        
        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in files:
                    if filename.endswith(".json"):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################

    def parse_article(self, raw_line: str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """

        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ""
        for sec in article["sections"]:
            txt_secs += sec["name"] + "\n" + sec["text"] + "\n"
            txt_secs += (
                "\n".join(
                    subsec["name"] + "\n" + subsec["text"] + "\n"
                    for subsec in sec["subsections"]
                )
                + "\n\n"
            )
            sec_names.append(sec["name"])
            sec_names.extend(subsec["name"] for subsec in sec["subsections"])
        article.pop("sections")  # no la necesitamos
        article["all"] = (
            article["title"] + "\n\n" + article["summary"] + "\n\n" + txt_secs
        )
        article["section-name"] = "\n".join(sec_names)

        return article

    def index_file(self, filename: str):
        """

        Indexa el contenido de un fichero.

        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.multifield y self.positional se debe ampliar el indexado


        """
        print(f"Indexing {filename}...")
        self.docs[len(self.docs)] = filename
        for i, line in enumerate(open(filename)):
            j = self.parse_article(line)
            if self.already_in_index(j):
                continue
            if j["url"] not in self.urls:
                self.urls.add(j["url"])
            artid = len(self.articles)
            self.articles[artid] = (len(self.docs) - 1, i + 1)
            pos = 0
            for field, tokenize in self.fields:
                if not self.multifield: 
                    field = "all"
                pos = 0
                if tokenize:
                    for token in self.tokenize(j[field]):
                        if token not in self.index[field]:
                            self.index[field][token] = {}
                        if artid not in self.index[field][token]:
                            self.index[field][token][artid] = []
                        self.index[field][token][artid].append(pos)
                        pos += 1
                    if not self.multifield: 
                        break
                else:
                    for url in j[field].split():
                        self.index[field][url] = artid
        
        if self.stemming:
            self.make_stemming()
        
        # En la version basica solo se debe indexar el contenido "article"
        #

    def set_stemming(self, v: bool):
        """

        Cambia el modo de stemming por defecto.

        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v

    def tokenize(self, text: str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(" ", text.lower()).split()

    def make_stemming(self):
        """

        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE STEMMING.

        "self.stemmer.stem(token) devuelve el stem del token"


        """
        
        for field, tokenize in self.fields: # Recorre los campos
            self.sindex[field] = {} # Inicializa el índice de stems de cada campo

        for field, tokenize in self.fields:
            for claveIndex in self.index[field]:
                stemTerm = self.stemmer.stem(claveIndex)
                if stemTerm not in self.sindex[field]:  # Si el stem de la entrada del "index" no está en el índice de stems
                    self.sindex[field][stemTerm] = []   # Se añade dicho stem como una nueva clave (inicialmente, una lista vacía)
                    self.sindex[field][stemTerm].append(claveIndex)  # Se añade a lista de valores para ese stem, la palabra de origen del "index"
                else:  # En caso de que el stem de la entrada del "index" ya estuviera en el índice de stems
                    if claveIndex not in self.sindex[field][stemTerm]:  # Se comprueba si la palabra de origen del "index" no estaba ya en la lista correspondiente a su stem
                        self.sindex[field][stemTerm].append(claveIndex)  # y, si no lo estaba, se añade
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

    def make_permuterm(self):
        """

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE PERMUTERM


        """
        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """
        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        print("===================================================")
        print(f"Number of indexed files: {len(self.docs)}")
        print("---------------------------------------------------")
        print(f"Number of indexed articles: {len(self.urls)}")
        print("---------------------------------------------------")
        print(f"TOKENS: \n         ")
        for field, tokenize in self.fields: 
            if len(self.index[field]) > 0:
                print(f"        # of tokens in '{field}': {len(self.index[field])}")
        if self.stemming:
            print("---------------------------------------------------")
            print(f"STEMS: \n         ")
            for field, tokenize in self.fields: 
                if len(self.sindex[field]) > 0:
                    print(f"        # of stems in '{field}': {len(self.sindex[field])}")
        print("===================================================")
        if self.positional:
            print("Positional queries are allowed.")
        else:
            print("Positional queries are NOT allowed.")
        print("========================================")

    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################

    def solve_query(self, query: str, prev: Dict = {}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """
        if query is None or len(query) == 0:
            return []
        
        def get_posting_list(term, field="all"):
            if isinstance(term, list):
                return self.get_positionals(term, field)
            else:
                return self.get_stemming(term, field) if self.use_stemming else self.get_posting(term, field)

        def apply_operator(values, operator):
            if operator == 'NOT':
                if values:
                    value = values.pop()
                    values.append(self.reverse_posting(value))
            else:
                if len(values) >= 2:
                    right = values.pop()
                    left = values.pop()
                    if operator == 'AND':
                        values.append(self.and_posting(left, right))
                    elif operator == 'OR':
                        values.append(self.or_posting(left, right))


        def eval_expression(tokens):
            values = []
            operators = []

            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token == 'NOT':
                    operators.append(token)
                elif token in {'AND', 'OR'}:
                    while operators and operators[-1] != '(':
                        apply_operator(values, operators.pop())
                    operators.append(token)
                elif token == '(':
                    operators.append(token)
                elif token == ')':
                    while operators and operators[-1] != '(':
                        apply_operator(values, operators.pop())
                    operators.pop()  # Sacar ( de los operadores
                else:
                    field = "all"
                    pos_term = []
                    while i < len(tokens) and tokens[i] not in {'AND', 'OR', 'NOT', '(', ')'}:
                        if ":" in tokens[i]:
                            field, term = tokens[i].split(":")
                            if len(term) > 0:
                                pos_term.append(term.lower())
                        else:
                            pos_term.append(tokens[i].lower())
                        i += 1
                    if len(pos_term) > 1:          
                        values.append(get_posting_list(pos_term, field))
                    else:
                        values.append(get_posting_list(pos_term[0], field))
                    continue  # Continue para evitar que se incemente i dos veces
                i += 1

            while operators:
                apply_operator(values, operators.pop())

            return values[0]

        query = re.sub(r'([()])', r' \1 ', query)
        query = query.replace('"', " ")
        terms = query.split()

        return eval_expression(terms)

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    def get_posting(self, term: str, field: Optional[str] = "all"):
        """

        Devuelve la posting list asociada a un termino.
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list

        NECESARIO PARA TODAS LAS VERSIONES

        """
        if term in self.index[field]:
            return list(self.index[field][term].keys())
        else:
            return []
        

    def get_positionals(self, terms: str, field: Optional[str] = "all"):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        res = []
        if terms[0] in self.index[field]:
            for tupla in self.index[field][terms[0]].items():
                artid, listpos = tupla
                # Para cada posicion en cada artículo compruebo si hay un termino en la pos + 1 en el mismo artículo
                for pos in listpos:
                    sigo = True
                    for term in (term for term in terms[1:] if sigo):
                        if term in self.index[field]:
                            if artid in self.index[field][term]:
                                if (pos + 1) in self.index[field][term][artid]:
                                    pos += 1
                                else:
                                    sigo = False
                            else:
                                sigo = False
                        else:
                            sigo = False
                    # En caso de que sigo sea true significa que lo ha encontrado, y si no esta repetido el artid lo guarda y pasa al siguiente artid
                    if sigo and artid not in res:
                        res += [artid]
                        break
            return res
        else:
            return []

        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################

    def get_stemming(self, term: str, field: Optional[str] = "all"):
        """

        Devuelve la posting list asociada al stem de un termino.
        NECESARIO PARA LA AMPLIACION DE STEMMING

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        
        stem = self.stemmer.stem(term)
        if stem in self.sindex[field]:
            postingList = []
            for word in list(self.sindex[field][stem]):
                postingList.extend(self.get_posting(word, field))
            return list(set(postingList))
        else:
            return []

        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

    def get_permuterm(self, term: str, field: Optional[str] = None):
        """

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        ##################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA PERMUTERM ##
        ##################################################
        pass

    def reverse_posting(self, p: list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        return [i for i in range(len(self.articles)) if i not in p]

    def and_posting(self, p1: list, p2: list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        p1, p2 = sorted(p1), sorted(p2)
        res = []
        i = 0
        j = 0

        # Meto lo artículos que coincidan en p1 y p2 de manera ordenada
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                res.append(p1[i])
                i += 1
                j += 1
            elif p1[i] <= p2[j]:
                i += 1
            elif p1[i] >= p2[j]:
                j += 1

        return res

    def or_posting(self, p1: list, p2: list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 o p2

        """
        p1, p2 = sorted(p1), sorted(p2)
        res = []
        i = 0
        j = 0

        # Meto todo artículo que haya en p1 y p2 en res de manera ordenada
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                res.append(p1[i])
                i += 1
                j += 1
            elif p1[i] <= p2[j]:
                res.append(p1[i])
                i += 1
            elif p1[i] >= p2[j]:
                res.append(p2[j])
                j += 1
        # Termino de recorrer las listas en caso de que una se mas larga que la otra
        for pos in range(i, len(p1)):
            res.append(p1[pos])

        for pos in range(j, len(p2)):
            res.append(p2[pos])

        return res

    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """

        pass
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################

    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql: List[str], verbose: bool = True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != "#":
                r = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f"{query}\t{len(r)}")
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results

    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result = len(self.solve_query(query))
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True

            else:
                print(line)

        return not errors

    def solve_and_show(self, query: str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        results = self.solve_query(query)                 
        i = 0
        stop = (
            len(results)
            if self.show_all
            else (10 if len(results) > 10 else len(results))
        )     
        while i < stop:
            articleFile = self.docs[self.articles[results[i]][0]]
            fp = open(articleFile)
            for j, line in enumerate(fp):
                if j == self.articles[results[i]][1]-1:
                    parsedArticle = self.parse_article(line)
            url = parsedArticle["url"]
            title = parsedArticle["title"]
            snippet = self.generate_snippet(query, parsedArticle["all"])        
            print(f"\n{i+1}. ID Articulo - {results[i]} URL: {url}")
            print(f"Titulo: {title}")
            if self.show_snippet:
                print("Snippet:")
                print(snippet)
            i += 1
        print("===================================================")
        print("Number of results: ", len(results))

        return len(results)
    
    def generate_snippet(self, query: str, doc: str, context_size: int = 5):
        """
        Genera un snippet para un documento basado en los términos de la consulta.

        :param query: La consulta original.
        :param doc: El texto completo del documento.
        :param context_size: El número de palabras de contexto a mostrar alrededor del término.
        :return: Un snippet del documento.
        """
        terms = query.split()
        words = doc.split()
        positions = []
        
        # Encontrar la primera ocurrencia de cada término
        for term in terms:
            try:
                pos = words.index(term)
                positions.append(pos)
            except ValueError:
                continue
        
        # Generar snippets
        snippet_parts = []
        for pos in positions:
            start = max(0, pos - context_size)
            end = min(len(words), pos + context_size + 1)
            snippet_parts.append(" ".join(words[start:end]))

        # Unir snippets y añadir delimitadores
        snippet = " ... ".join(snippet_parts)
        snippet = "..." + snippet + "..."
        return snippet
