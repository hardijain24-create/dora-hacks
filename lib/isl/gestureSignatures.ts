export const ISL_GESTURE_SIGNATURES = {

  // ============================================================
  // GREETINGS
  // ============================================================

  HELLO: {
    category: "greeting",
    hands: 1,

    handshape: [
      "open_palm",
      "flat_hand",
      "fingers_extended"
    ],

    location: [
      "forehead",
      "temple",
      "upper_face"
    ],

    orientation: [
      "palm_forward",
      "palm_slightly_outward"
    ],

    movement: [
      "short_outward",
      "away_from_head"
    ],

    temporal: {
      dynamic: true,
      repetitions: 1
    }
  },


  THANK_YOU: {
    category: "greeting",
    hands: 1,

    handshape: [
      "open_palm",
      "flat_hand",
      "fingers_extended"
    ],

    location: [
      "chin",
      "lower_face"
    ],

    movement: [
      "chin_to_outward",
      "forward"
    ],

    temporal: {
      dynamic: true,
      repetitions: 1
    }
  },


  WELCOME: {
    category: "greeting",
    hands: 1,

    handshape: [
      "open_palm"
    ],

    movement: [
      "outward",
      "presenting"
    ],

    temporal: {
      dynamic: true
    }
  },


  // ============================================================
  // INTRODUCTION / IDENTITY
  // ============================================================

  NAME: {
    category: "introduction",
    hands: 2,

    handshape: [
      "index_extended",
      "selected_fingers"
    ],

    relationship: [
      "hands_interact"
    ],

    movement: [
      "small_repeated"
    ],

    temporal: {
      dynamic: true,
      repetitions: 2
    }
  },


  I_ME: {
    category: "person",
    hands: 1,

    handshape: [
      "index_extended"
    ],

    location: [
      "self",
      "chest"
    ],

    movement: [
      "toward_self"
    ]
  },


  YOU: {
    category: "person",
    hands: 1,

    handshape: [
      "index_extended"
    ],

    movement: [
      "point_outward"
    ],

    direction: "away_from_signer"
  },


  FRIEND: {
    category: "person",
    hands: 2,

    handshape: [
      "curved_or_hooked_fingers"
    ],

    relationship: [
      "hands_interlock_or_contact"
    ],

    movement: [
      "interacting"
    ]
  },


  // ============================================================
  // PLACES
  // ============================================================

  HOUSE: {
    category: "place",

    hands: 2,

    handshape: [
      "open_or_flat"
    ],

    relationship: [
      "hands_form_roof_geometry"
    ],

    movement: [
      "hands_approach",
      "roof_shape"
    ],

    geometry: {
      bilateral: true,
      apex_like_structure: true
    }
  },


  UNIVERSITY: {
    category: "education_place",

    hands: 1,

    handshape: [
      "varies"
    ],

    location: [
      "neutral_space",
      "upper_body"
    ],

    movement: [
      "dynamic"
    ],

    note:
      "Do not classify from handshape alone. Prefer INCLUDE temporal model."
  },


  SCHOOL: {
    category: "education_place",

    hands: 2,

    handshape: [
      "flat_hand"
    ],

    relationship: [
      "hands_contact_or_interact"
    ],

    movement: [
      "repeated_contact"
    ]
  },


  HOSPITAL: {
    category: "medical_place",

    hands: 1,

    handshape: [
      "configured_hand"
    ],

    location: [
      "upper_body"
    ],

    movement: [
      "dynamic"
    ],

    note:
      "Use model prediction as primary signal."
  },


  BANK: {
    category: "place",

    hands: 1,

    handshape: [
      "configured_hand"
    ],

    movement: [
      "directional"
    ],

    note:
      "Temporal signature required."
  },


  HOTEL: {
    category: "place",

    hands: 1,

    handshape: [
      "configured_hand"
    ],

    movement: [
      "dynamic"
    ],

    note:
      "Do not use generic open-palm classification."
  },


  RESTAURANT: {
    category: "place",

    hands: 1,

    handshape: [
      "configured_hand"
    ],

    location: [
      "mouth_or_upper_body"
    ],

    movement: [
      "toward_mouth"
    ]
  },


  OFFICE: {
    category: "place",

    hands: 1,

    handshape: [
      "configured_hand"
    ],

    movement: [
      "dynamic"
    ]
  },


  MARKET: {
    category: "place",

    hands: 2,

    handshape: [
      "open_or_configured"
    ],

    movement: [
      "interaction",
      "repeated"
    ]
  },


  CITY: {
    category: "place",

    hands: 1,

    handshape: [
      "configured_hand"
    ],

    movement: [
      "repeated_or_directional"
    ]
  },


  LOCATION: {
    category: "place",

    hands: 1,

    handshape: [
      "index_or_configured"
    ],

    movement: [
      "directional"
    ]
  },


  WINDOW: {
    category: "object/place",

    hands: 2,

    handshape: [
      "open_or_flat"
    ],

    relationship: [
      "bilateral_frame_geometry"
    ],

    movement: [
      "open_close",
      "frame_like"
    ],

    note:
      "Requires temporal + bilateral geometry; NEVER index-finger-only."
  },


  // ============================================================
  // COLORS
  // ============================================================

  RED: {
    category: "color",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "localized"
    ],

    note:
      "Color signs can have regional variants. Use INCLUDE model first."
  },


  BLUE: {
    category: "color",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "repeated_or_dynamic"
    ]
  },


  GREEN: {
    category: "color",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "dynamic"
    ]
  },


  YELLOW: {
    category: "color",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "dynamic"
    ],

    note:
      "INCLUDE class is the authoritative classifier."
  },


  BLACK: {
    category: "color",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "dynamic"
    ]
  },


  WHITE: {
    category: "color",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "dynamic"
    ]
  },


  ORANGE: {
    category: "color",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "repeated"
    ]
  },


  PINK: {
    category: "color",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "dynamic"
    ]
  },


  PURPLE: {
    category: "color",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "dynamic"
    ]
  },


  BROWN: {
    category: "color",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "dynamic"
    ]
  },


  // ============================================================
  // ANIMALS
  // ============================================================

  COW: {
    category: "animal",
    hands: 1,

    handshape: [
      "animal_specific"
    ],

    location: [
      "head_or_neutral_space"
    ],

    movement: [
      "dynamic"
    ],

    temporal: {
      dynamic: true
    },

    CRITICAL:
      "Never classify Cow from open palm alone."
  },


  DOG: {
    category: "animal",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "short_repeated"
    ]
  },


  CAT: {
    category: "animal",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "pulling_or_curving"
    ]
  },


  HORSE: {
    category: "animal",
    hands: 1,

    handshape: [
      "configured"
    ],

    location: [
      "side_of_head"
    ],

    movement: [
      "repeated"
    ]
  },


  BIRD: {
    category: "animal",
    hands: 1,

    handshape: [
      "configured",
      "beak_like"
    ],

    location: [
      "face"
    ],

    movement: [
      "opening_closing"
    ]
  },


  FISH: {
    category: "animal",
    hands: 1,

    handshape: [
      "flat_hand"
    ],

    movement: [
      "swimming_motion"
    ]
  },


  ELEPHANT: {
    category: "animal",
    hands: 1,

    handshape: [
      "configured"
    ],

    location: [
      "face"
    ],

    movement: [
      "trunk_like"
    ]
  },


  LION: {
    category: "animal",
    hands: 1,

    handshape: [
      "configured"
    ],

    location: [
      "head"
    ],

    movement: [
      "mane_like"
    ]
  },


  TIGER: {
    category: "animal",
    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "face_related"
    ]
  },


  MONKEY: {
    category: "animal",
    hands: 2,

    handshape: [
      "configured"
    ],

    movement: [
      "body_related",
      "repeated"
    ]
  },


  // ============================================================
  // CLOTHING
  // ============================================================

  SHIRT: {
    category: "clothing",
    hands: 2,

    handshape: [
      "configured"
    ],

    location: [
      "upper_body"
    ],

    movement: [
      "clothing_related"
    ]
  },


  PANT: {
    category: "clothing",
    hands: 2,

    handshape: [
      "configured"
    ],

    location: [
      "lower_body"
    ],

    movement: [
      "downward_or_clothing_related"
    ]
  },


  DRESS: {
    category: "clothing",
    hands: 2,

    handshape: [
      "open_or_configured"
    ],

    location: [
      "torso"
    ],

    movement: [
      "downward"
    ]
  },


  SAREE: {
    category: "clothing",
    hands: 2,

    handshape: [
      "configured"
    ],

    location: [
      "torso"
    ],

    movement: [
      "wrapping_or_side_motion"
    ]
  },


  SKIRT: {
    category: "clothing",
    hands: 2,

    location: [
      "waist",
      "lower_body"
    ],

    movement: [
      "side_or_downward"
    ]
  },


  T_SHIRT: {
    category: "clothing",

    hands: 2,

    location: [
      "upper_body"
    ],

    movement: [
      "clothing_related"
    ]
  },


  COAT: {
    category: "clothing",

    hands: 2,

    location: [
      "upper_body"
    ],

    movement: [
      "front_or_side"
    ]
  },


  JACKET: {
    category: "clothing",

    hands: 2,

    location: [
      "upper_body"
    ],

    movement: [
      "clothing_related"
    ]
  },


  HAT: {
    category: "clothing",

    hands: 1,

    location: [
      "head"
    ],

    movement: [
      "head_related"
    ]
  },


  CAP: {
    category: "clothing",

    hands: 1,

    location: [
      "head"
    ],

    movement: [
      "head_related"
    ]
  },


  SOCKS: {
    category: "clothing",

    hands: 2,

    location: [
      "lower_body"
    ]
  },


  // ============================================================
  // FOOTWEAR
  // ============================================================

  SHOE: {
    category: "footwear",

    hands: 2,

    handshape: [
      "configured"
    ],

    relationship: [
      "hands_oriented_as_object"
    ],

    movement: [
      "object_related"
    ]
  },


  SLIPPER: {
    category: "footwear",

    hands: 2,

    handshape: [
      "configured"
    ],

    movement: [
      "object_related"
    ]
  },


  SANDAL: {
    category: "footwear",

    hands: 2,

    handshape: [
      "configured"
    ],

    movement: [
      "object_related"
    ]
  },


  // ============================================================
  // MEDICAL
  // ============================================================

  DOCTOR: {
    category: "medical",

    hands: 1,

    handshape: [
      "configured"
    ],

    location: [
      "upper_body"
    ],

    movement: [
      "dynamic"
    ],

    note:
      "Model prediction is primary; heuristic only validates temporal consistency."
  },


  NURSE: {
    category: "medical",

    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "dynamic"
    ]
  },


  PATIENT: {
    category: "medical",

    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "dynamic"
    ]
  },


  MEDICINE: {
    category: "medical",

    hands: 1,

    handshape: [
      "configured"
    ],

    movement: [
      "object_related"
    ]
  },


  // ============================================================
  // GENERIC HIGH-VALUE HANDSHAPE FEATURES
  // ============================================================

  OPEN_PALM: {
    feature_only: true,

    handshape: [
      "all_fingers_extended",
      "fingers_separated_or_extended"
    ],

    orientation: [
      "palm_forward",
      "palm_sideways",
      "palm_down"
    ],

    WARNING:
      "NOT a semantic ISL word. Never directly map Open Palm → Hello/Cow/etc."
  },


  CLOSED_FIST: {
    feature_only: true,

    handshape: [
      "all_fingers_flexed"
    ],

    WARNING:
      "NOT a semantic ISL word."
  },


  INDEX_POINT: {
    feature_only: true,

    handshape: [
      "index_extended",
      "other_fingers_flexed"
    ],

    WARNING:
      "NOT a semantic ISL word."
  },


  TWO_FINGERS: {
    feature_only: true,

    handshape: [
      "index_middle_extended"
    ],

    WARNING:
      "NOT a semantic ISL word."
  },


  THREE_FINGERS: {
    feature_only: true,

    handshape: [
      "three_selected_fingers_extended"
    ],

    WARNING:
      "NOT a semantic ISL word."
  },


  FOUR_FINGERS: {
    feature_only: true,

    handshape: [
      "four_selected_fingers_extended"
    ],

    WARNING:
      "NOT a semantic ISL word."
  },


  FIVE_FINGERS: {
    feature_only: true,

    handshape: [
      "all_fingers_extended"
    ],

    WARNING:
      "NOT a semantic ISL word."
  },


  THUMBS_UP: {
    feature_only: true,

    handshape: [
      "thumb_extended",
      "other_fingers_flexed"
    ],

    WARNING:
      "Do not automatically map this to YES in ISL."
  },


  OPEN_CLOSE: {
    feature_only: true,

    temporal: {
      transition: "open_to_closed",
      transition_count: 1
    },

    WARNING:
      "Temporal feature, not a semantic sign."
  },


  WAVE: {
    feature_only: true,

    handshape: [
      "open_palm"
    ],

    movement: [
      "side_to_side"
    ],

    temporal: {
      repeated: true
    },

    WARNING:
      "A wave is not automatically the ISL sign for Hello."
  }

} as const;

export const ISL_GESTURE_SIGNATURES_2 = {

  // ============================================================
  // BASIC COMMUNICATION
  // ============================================================

  YES: {
    category: "communication",
    hands: 1,
    handshape: ["closed_fist"],
    orientation: ["neutral"],
    movement: ["small_repeated_downward"],
    temporal: { dynamic: true, repetitions: 2 },
    note: "Candidate heuristic; verify against ISL video."
  },

  NO: {
    category: "communication",
    hands: 1,
    handshape: ["index_middle_extended"],
    movement: ["closing_toward_thumb"],
    temporal: { dynamic: true },
    note: "Candidate heuristic; verify against ISL video."
  },

  PLEASE: {
    category: "communication",
    hands: 1,
    handshape: ["open_palm"],
    location: ["chest"],
    movement: ["circular"],
    temporal: { dynamic: true }
  },

  SORRY: {
    category: "communication",
    hands: 1,
    handshape: ["closed_fist"],
    location: ["chest"],
    movement: ["circular"],
    temporal: { dynamic: true }
  },

  HELP: {
    category: "communication",
    hands: 2,
    handshape: ["open_palm", "configured_hand"],
    relationship: ["one_hand_supports_other"],
    movement: ["upward"],
    temporal: { dynamic: true }
  },

  STOP: {
    category: "communication",
    hands: 1,
    handshape: ["open_palm"],
    orientation: ["palm_forward"],
    movement: ["forward"],
    temporal: { dynamic: true }
  },

  WAIT: {
    category: "communication",
    hands: 2,
    handshape: ["open_or_curved"],
    movement: ["hold"],
    temporal: { static_hold: true }
  },

  FINISH: {
    category: "communication",
    hands: 2,
    handshape: ["open_palm"],
    movement: ["outward"],
    temporal: { dynamic: true }
  },

  AGAIN: {
    category: "communication",
    hands: 1,
    handshape: ["configured"],
    movement: ["repeated"],
    temporal: { repetitions: 2 }
  },

  UNDERSTAND: {
    category: "communication",
    hands: 1,
    handshape: ["configured"],
    location: ["forehead"],
    movement: ["outward"],
    temporal: { dynamic: true }
  },


  // ============================================================
  // PEOPLE / FAMILY
  // ============================================================

  MAN: {
    category: "person",
    hands: 1,
    handshape: ["configured"],
    location: ["head_or_upper_body"],
    movement: ["directional"]
  },

  WOMAN: {
    category: "person",
    hands: 1,
    handshape: ["configured"],
    location: ["face_or_upper_body"],
    movement: ["directional"]
  },

  CHILD: {
    category: "person",
    hands: 2,
    handshape: ["open_palm"],
    location: ["lower_body"],
    movement: ["height_related"]
  },

  BABY: {
    category: "person",
    hands: 2,
    handshape: ["curved_hands"],
    location: ["front_body"],
    movement: ["rocking"],
    temporal: { dynamic: true }
  },

  MOTHER: {
    category: "family",
    hands: 1,
    handshape: ["open_or_configured"],
    location: ["head_or_face"],
    movement: ["localized"]
  },

  FATHER: {
    category: "family",
    hands: 1,
    handshape: ["open_or_configured"],
    location: ["forehead"],
    movement: ["localized"]
  },

  BROTHER: {
    category: "family",
    hands: 2,
    handshape: ["configured"],
    relationship: ["hands_interact"],
    movement: ["contact_or_directional"]
  },

  SISTER: {
    category: "family",
    hands: 2,
    handshape: ["configured"],
    relationship: ["hands_interact"],
    movement: ["contact_or_directional"]
  },

  FAMILY: {
    category: "family",
    hands: 2,
    handshape: ["open_or_curved"],
    movement: ["circular_or_grouping"],
    temporal: { dynamic: true }
  },

  FRIEND: {
    category: "social",
    hands: 2,
    handshape: ["hooked_or_curved"],
    relationship: ["hands_interact"],
    movement: ["interlocking"]
  },


  // ============================================================
  // EMOTIONS
  // ============================================================

  HAPPY: {
    category: "emotion",
    hands: 2,
    handshape: ["open_palm"],
    location: ["chest"],
    movement: ["upward_repeated"],
    temporal: { repetitions: 2 }
  },

  SAD: {
    category: "emotion",
    hands: 2,
    handshape: ["open_or_curved"],
    location: ["face"],
    movement: ["downward"],
    temporal: { dynamic: true }
  },

  ANGRY: {
    category: "emotion",
    hands: 2,
    handshape: ["configured"],
    location: ["face_or_chest"],
    movement: ["outward_or_forceful"],
    temporal: { dynamic: true }
  },

  LOVE: {
    category: "emotion",
    hands: 2,
    handshape: ["closed_or_curved"],
    location: ["chest"],
    movement: ["toward_self"],
    temporal: { dynamic: true }
  },

  LIKE: {
    category: "emotion",
    hands: 1,
    handshape: ["configured"],
    location: ["chest"],
    movement: ["toward_self"],
    temporal: { dynamic: true }
  },

  DISLIKE: {
    category: "emotion",
    hands: 1,
    handshape: ["configured"],
    location: ["chest"],
    movement: ["away_from_self"],
    temporal: { dynamic: true }
  },

  GOOD: {
    category: "evaluation",
    hands: 1,
    handshape: ["configured"],
    location: ["mouth_or_upper_body"],
    movement: ["outward"],
    temporal: { dynamic: true }
  },

  BAD: {
    category: "evaluation",
    hands: 1,
    handshape: ["configured"],
    movement: ["downward_or_away"],
    temporal: { dynamic: true }
  },

  BEAUTIFUL: {
    category: "description",
    hands: 1,
    handshape: ["open_or_configured"],
    location: ["face"],
    movement: ["outward"],
    temporal: { dynamic: true }
  },

  IMPORTANT: {
    category: "description",
    hands: 1,
    handshape: ["configured"],
    movement: ["emphatic"],
    temporal: { dynamic: true }
  },


  // ============================================================
  // TIME
  // ============================================================

  TODAY: {
    category: "time",
    hands: 2,
    handshape: ["configured"],
    location: ["neutral_space"],
    movement: ["toward_current_position"]
  },

  TOMORROW: {
    category: "time",
    hands: 1,
    handshape: ["configured"],
    movement: ["forward"],
    temporal: { dynamic: true }
  },

  YESTERDAY: {
    category: "time",
    hands: 1,
    handshape: ["configured"],
    movement: ["backward"],
    temporal: { dynamic: true }
  },

  MORNING: {
    category: "time",
    hands: 1,
    handshape: ["configured"],
    movement: ["rising"],
    temporal: { dynamic: true }
  },

  NIGHT: {
    category: "time",
    hands: 1,
    handshape: ["open_or_configured"],
    movement: ["downward_or_covering"]
  },

  DAY: {
    category: "time",
    hands: 1,
    handshape: ["configured"],
    movement: ["arc_or_directional"],
    temporal: { dynamic: true }
  },

  MONTH: {
    category: "time",
    hands: 1,
    handshape: ["configured"],
    movement: ["cyclic_or_directional"]
  },

  YEAR: {
    category: "time",
    hands: 2,
    handshape: ["configured"],
    relationship: ["hands_interact"],
    movement: ["cyclic"]
  },


  // ============================================================
  // TRANSPORT
  // ============================================================

  CAR: {
    category: "transport",
    hands: 2,
    handshape: ["closed_or_configured"],
    relationship: ["hands_parallel"],
    movement: ["steering"],
    temporal: { dynamic: true }
  },

  BUS: {
    category: "transport",
    hands: 2,
    handshape: ["configured"],
    movement: ["large_directional"],
    temporal: { dynamic: true }
  },

  TRAIN: {
    category: "transport",
    hands: 2,
    handshape: ["configured"],
    relationship: ["parallel"],
    movement: ["forward"],
    temporal: { dynamic: true }
  },

  BICYCLE: {
    category: "transport",
    hands: 2,
    handshape: ["configured"],
    movement: ["circular_alternating"],
    temporal: { dynamic: true }
  },

  MOTORCYCLE: {
    category: "transport",
    hands: 2,
    handshape: ["configured"],
    movement: ["steering"],
    temporal: { dynamic: true }
  },

  WALK: {
    category: "transport_action",
    hands: 2,
    handshape: ["configured"],
    movement: ["alternating"],
    temporal: { repetitive: true }
  },

  RUN: {
    category: "transport_action",
    hands: 2,
    handshape: ["configured"],
    movement: ["fast_alternating"],
    temporal: { repetitive: true }
  },


  // ============================================================
  // EDUCATION
  // ============================================================

  STUDENT: {
    category: "education",
    hands: 2,
    handshape: ["configured"],
    location: ["upper_body"],
    movement: ["study_related"]
  },

  TEACHER: {
    category: "education",
    hands: 2,
    handshape: ["configured"],
    movement: ["instructional"],
    temporal: { dynamic: true }
  },

  BOOK: {
    category: "education",
    hands: 2,
    handshape: ["flat_hands"],
    relationship: ["palms_together"],
    movement: ["opening_like_book"],
    temporal: { dynamic: true }
  },

  PEN: {
    category: "education",
    hands: 1,
    handshape: ["pinch"],
    movement: ["writing"],
    temporal: { dynamic: true }
  },

  PENCIL: {
    category: "education",
    hands: 1,
    handshape: ["pinch_or_configured"],
    movement: ["writing"],
    temporal: { dynamic: true }
  },

  PAPER: {
    category: "education",
    hands: 2,
    handshape: ["flat_hands"],
    relationship: ["parallel"],
    movement: ["flat_or_open"]
  },


  // ============================================================
  // TECHNOLOGY
  // ============================================================

  MOBILE: {
    category: "technology",
    hands: 1,
    handshape: ["phone_like_configuration"],
    location: ["ear_or_face"],
    movement: ["phone_related"]
  },

  COMPUTER: {
    category: "technology",
    hands: 2,
    handshape: ["configured"],
    location: ["front_body"],
    movement: ["typing"],
    temporal: { repetitive: true }
  },

  TELEVISION: {
    category: "technology",
    hands: 2,
    handshape: ["configured"],
    movement: ["frame_or_screen_related"]
  },

  CAMERA: {
    category: "technology",
    hands: 2,
    handshape: ["frame_like"],
    location: ["face"],
    movement: ["viewing_or_capture"]
  },


  // ============================================================
  // OBJECTS
  // ============================================================

  CHAIR: {
    category: "object",
    hands: 2,
    handshape: ["configured"],
    movement: ["object_shape"]
  },

  TABLE: {
    category: "object",
    hands: 2,
    handshape: ["flat_hands"],
    relationship: ["parallel"],
    movement: ["horizontal"]
  },

  DOOR: {
    category: "object",
    hands: 1,
    handshape: ["configured"],
    movement: ["open_close"],
    temporal: { dynamic: true }
  },

  KEY: {
    category: "object",
    hands: 1,
    handshape: ["pinch_or_configured"],
    movement: ["turning"],
    temporal: { dynamic: true }
  },

  PHONE: {
    category: "object",
    hands: 1,
    handshape: ["phone_like_configuration"],
    location: ["ear"],
    movement: ["communication_related"]
  },

  MONEY: {
    category: "finance",
    hands: 1,
    handshape: ["configured"],
    movement: ["rubbing_or_exchange"],
    temporal: { dynamic: true }
  },

  FOOD: {
    category: "daily_life",
    hands: 1,
    handshape: ["configured"],
    location: ["mouth"],
    movement: ["toward_mouth"],
    temporal: { dynamic: true }
  },

  WATER: {
    category: "daily_life",
    hands: 1,
    handshape: ["configured"],
    location: ["mouth_or_chin"],
    movement: ["drink_related"],
    temporal: { dynamic: true }
  },

  MEDICINE: {
    category: "medical",
    hands: 1,
    handshape: ["configured"],
    movement: ["object_related"],
    temporal: { dynamic: true }
  },

  // ============================================================
  // ACTIONS
  // ============================================================

  COME: {
    category: "action",
    hands: 1,
    handshape: ["open_or_configured"],
    movement: ["toward_signer"],
    temporal: { dynamic: true }
  },

  GO: {
    category: "action",
    hands: 1,
    handshape: ["configured"],
    movement: ["away_from_signer"],
    temporal: { dynamic: true }
  },

  GIVE: {
    category: "action",
    hands: 1,
    handshape: ["open_or_configured"],
    movement: ["toward_other"],
    temporal: { dynamic: true }
  },

  TAKE: {
    category: "action",
    hands: 1,
    handshape: ["configured"],
    movement: ["toward_self"],
    temporal: { dynamic: true }
  },

  EAT: {
    category: "action",
    hands: 1,
    handshape: ["configured"],
    location: ["mouth"],
    movement: ["toward_mouth"],
    temporal: { repetitive: true }
  },

  DRINK: {
    category: "action",
    hands: 1,
    handshape: ["cup_like"],
    location: ["mouth"],
    movement: ["tilting_toward_mouth"],
    temporal: { dynamic: true }
  },

  SLEEP: {
    category: "action",
    hands: 1,
    handshape: ["open_or_flat"],
    location: ["face"],
    movement: ["downward"],
    temporal: { dynamic: true }
  },

  SIT: {
    category: "action",
    hands: 2,
    handshape: ["configured"],
    movement: ["downward"],
    temporal: { dynamic: true }
  },

  STAND: {
    category: "action",
    hands: 2,
    handshape: ["configured"],
    movement: ["upward"],
    temporal: { dynamic: true }
  },

} as const;
