class Params:
    def __init__(self):

        #self.path='/content/drive/My Drive/9517project/raymond/Data/Sequences/03/'     
        self.path = 'C:/Users/dell/Downloads/COMP9517 21T3 Group Project Dataset/Sequences/01'
        self.preprocess = "preprocess_1" 
            #type 1: "only_area_min"
            #type 2: "only_intensity_min"
            #type 3: "only_area_min_sec"
            #type 4:"0.5*area+0.5*intensity_min_sec"
            #type 5:"0.5*area+0.5*intensity_min"
        self.similarity_type ="0.5*area+0.5*intensity_min_sec"

        # Two trivial parameters added after demo
        self.cuda = False # Use GPU or not
        self.min_cell_area = 20 # The minimum cell area threshold
        self.LOC_2_1 = (20,10)
        self.LOC_2_2 = (20,30)
        self.LOC_2_3 = (20,50)
        self.LOC_2_4 = (20,70)

        self.CELL_DETAILS_LOC_1 = (20, 10)
        self.CELL_DETAILS_LOC_2 = (20, 30)
        self.CELL_DETAILS_LOC_3 = (20, 50)
        self.CELL_DETAILS_LOC_4 = (20, 70)
        