module Jekyll
  class ReferenceLink < Liquid::Block
    def initialize(tag_name, name, tokens)
      super
      @link_name = name.strip
    end

    def render(context)
      text = super
      "<a name=\"#{@link_name}\">[#{text}][#{@link_name}]</a>"
    end
  end
end

Liquid::Template.register_tag('reflink', Jekyll::ReferenceLink)
